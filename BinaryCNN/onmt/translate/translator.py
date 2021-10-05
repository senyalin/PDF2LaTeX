#!/usr/bin/env python
""" Translator Class and builder """
from __future__ import print_function
import codecs
import os
import math
import time
from itertools import count

import torch

import onmt.model_builder
import onmt.translate.beam
import onmt.inputters as inputters
import onmt.decoders.ensemble
from onmt.translate.beam_search import BeamSearch
from onmt.translate.random_sampling import RandomSampling
from onmt.utils.misc import tile, set_random_seed
from onmt.modules.copy_generator import collapse_copy_scores


def build_translator(opt, report_score=True, logger=None, out_file=None):
    if out_file is None:
        out_file = codecs.open(opt.output, 'w+', 'utf-8')

    load_test_model = onmt.decoders.ensemble.load_test_model \
        if len(opt.models) > 1 else onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = Translator.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        out_file=out_file,
        report_score=report_score,
        logger=logger
    )
    return translator


class Translator(object):
    """Translate a batch of sentences with a saved model.

    Args:
        model (onmt.modules.NMTModel): NMT model to use for translation
        fields (dict[str, torchtext.data.Field]): A dict
            mapping each side to its list of name-Field pairs.
        src_reader (onmt.inputters.DataReaderBase): Source reader.
        tgt_reader (onmt.inputters.TextDataReader): Target reader.
        gpu (int): GPU device. Set to negative for no GPU.
        n_best (int): How many beams to wait for.
        min_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        max_length (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        beam_size (int): Number of beams.
        random_sampling_topk (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        random_sampling_temp (int): See
            :class:`onmt.translate.random_sampling.RandomSampling`.
        stepwise_penalty (bool): Whether coverage penalty is applied every step
            or not.
        dump_beam (bool): Debugging option.
        block_ngram_repeat (int): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        ignore_when_blocking (set or frozenset): See
            :class:`onmt.translate.decode_strategy.DecodeStrategy`.
        replace_unk (bool): Replace unknown token.
        data_type (str): Source data type.
        verbose (bool): Print/log every translation.
        report_bleu (bool): Print/log Bleu metric.
        report_rouge (bool): Print/log Rouge metric.
        report_time (bool): Print/log total time/frequency.
        copy_attn (bool): Use copy attention.
        global_scorer (onmt.translate.GNMTGlobalScorer): Translation
            scoring/reranking object.
        out_file (TextIO or codecs.StreamReaderWriter): Output file.
        report_score (bool) : Whether to report scores
        logger (logging.Logger or NoneType): Logger.
    """

    def __init__(
            self,
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=-1,
            n_best=1,
            min_length=0,
            max_length=100,
            ratio=0.,
            beam_size=30,
            random_sampling_topk=1,
            random_sampling_temp=1,
            stepwise_penalty=None,
            dump_beam=False,
            block_ngram_repeat=0,
            ignore_when_blocking=frozenset(),
            replace_unk=False,
            phrase_table="",
            data_type="text",
            verbose=False,
            report_bleu=False,
            report_rouge=False,
            report_time=False,
            copy_attn=False,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None,
            seed=-1):
        self.model = model
        self.fields = fields
        tgt_field = dict(self.fields)["tgt"].base_field
        self._tgt_vocab = tgt_field.vocab
        self._tgt_eos_idx = self._tgt_vocab.stoi[tgt_field.eos_token]
        self._tgt_pad_idx = self._tgt_vocab.stoi[tgt_field.pad_token]
        self._tgt_bos_idx = self._tgt_vocab.stoi[tgt_field.init_token]
        self._tgt_unk_idx = self._tgt_vocab.stoi[tgt_field.unk_token]
        self._tgt_vocab_len = len(self._tgt_vocab)

        self._gpu = gpu
        self._use_cuda = gpu > -1
        self._dev = torch.device("cuda", self._gpu) \
            if self._use_cuda else torch.device("cpu")

        self.n_best = n_best
        self.max_length = max_length

        self.beam_size = beam_size
        self.random_sampling_temp = random_sampling_temp
        self.sample_from_topk = random_sampling_topk

        self.min_length = min_length
        self.ratio = ratio
        self.stepwise_penalty = stepwise_penalty
        self.dump_beam = dump_beam
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self._exclusion_idxs = {
            self._tgt_vocab.stoi[t] for t in self.ignore_when_blocking}
        self.src_reader = src_reader
        self.tgt_reader = tgt_reader
        self.replace_unk = replace_unk
        if self.replace_unk and not self.model.decoder.attentional:
            raise ValueError(
                "replace_unk requires an attentional decoder.")
        self.phrase_table = phrase_table
        self.data_type = data_type
        self.verbose = verbose
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.report_time = report_time

        self.copy_attn = copy_attn

        self.global_scorer = global_scorer
        if self.global_scorer.has_cov_pen and \
                not self.model.decoder.attentional:
            raise ValueError(
                "Coverage penalty requires an attentional decoder.")
        self.out_file = out_file
        self.report_score = report_score
        self.logger = logger

        self.use_filter_pred = False
        self._filter_pred = None

        # for debugging
        self.beam_trace = self.dump_beam != ""
        self.beam_accum = None
        if self.beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": [],
                "log_probs": []}

        set_random_seed(seed, self._use_cuda)

    @classmethod
    def from_opt(
            cls,
            model,
            fields,
            opt,
            model_opt,
            global_scorer=None,
            out_file=None,
            report_score=True,
            logger=None):
        """Alternate constructor.

        Args:
            model (onmt.modules.NMTModel): See :func:`__init__()`.
            fields (dict[str, torchtext.data.Field]): See
                :func:`__init__()`.
            opt (argparse.Namespace): Command line options
            model_opt (argparse.Namespace): Command line options saved with
                the model checkpoint.
            global_scorer (onmt.translate.GNMTGlobalScorer): See
                :func:`__init__()`..
            out_file (TextIO or codecs.StreamReaderWriter): See
                :func:`__init__()`.
            report_score (bool) : See :func:`__init__()`.
            logger (logging.Logger or NoneType): See :func:`__init__()`.
        """

        src_reader = inputters.str2reader[opt.data_type].from_opt(opt)
        tgt_reader = inputters.str2reader["text"].from_opt(opt)
        return cls(
            model,
            fields,
            src_reader,
            tgt_reader,
            gpu=opt.gpu,
            n_best=opt.n_best,
            min_length=opt.min_length,
            max_length=opt.max_length,
            ratio=opt.ratio,
            beam_size=opt.beam_size,
            random_sampling_topk=opt.random_sampling_topk,
            random_sampling_temp=opt.random_sampling_temp,
            stepwise_penalty=opt.stepwise_penalty,
            dump_beam=opt.dump_beam,
            block_ngram_repeat=opt.block_ngram_repeat,
            ignore_when_blocking=set(opt.ignore_when_blocking),
            replace_unk=opt.replace_unk,
            phrase_table=opt.phrase_table,
            data_type=opt.data_type,
            verbose=opt.verbose,
            report_bleu=opt.report_bleu,
            report_rouge=opt.report_rouge,
            report_time=opt.report_time,
            copy_attn=model_opt.copy_attn,
            global_scorer=global_scorer,
            out_file=out_file,
            report_score=report_score,
            logger=logger,
            seed=opt.seed)

    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def translate(
            self,
            src,
            tgt=None,
            src_dir=None,
            batch_size=None,
            attn_debug=False,
            phrase_table=""):
        """Translate content of ``src`` and get gold scores from ``tgt``.

        Args:
            src: See :func:`self.src_reader.read()`.
            tgt: See :func:`self.tgt_reader.read()`.
            src_dir: See :func:`self.src_reader.read()` (only relevant
                for certain types of data).
            batch_size (int): size of examples per mini-batch
            attn_debug (bool): enables the attention logging

        Returns:
            (`list`, `list`)

            * all_scores is a list of `batch_size` lists of `n_best` scores
            * all_predictions is a list of `batch_size` lists
                of `n_best` predictions
        """

        if batch_size is None:
            raise ValueError("batch_size must be set")

        data = inputters.Dataset(
            self.fields,
            readers=([self.src_reader, self.tgt_reader]
                     if tgt else [self.src_reader]),
            data=[("src", src), ("tgt", tgt)] if tgt else [("src", src)],
            dirs=[src_dir, None] if tgt else [src_dir],
            sort_key=inputters.str2sortkey[self.data_type],
            filter_pred=self._filter_pred
        )

        data_iter = inputters.OrderedIterator(
            dataset=data,
            device=self._dev,
            batch_size=batch_size,
            train=False,
            sort=False,
            sort_within_batch=True,
            shuffle=False
        )

        step = 0
        for batch in data_iter:
            outputs = self.translate_batch(
                batch, data.src_vocabs, attn_debug
            )
            # translations = xlation_builder.from_batch(batch_data)
            # pred = outputs.max(1)[1]

            softmax = torch.nn.Softmax()
            scores = softmax(outputs)
            translations = scores[:, 0]  # scores for being plaintext

            step += batch.batch_size
            # print('Processing step: ' + str(step))

            scores = [None] * batch.batch_size
            start_idx = torch.min(batch.indices).data.tolist()
            for i, trans in enumerate(translations):
                # recover the order of the keys
                score = "{0:.5f}".format(trans.data.tolist())
                idx = batch.indices[i].data.tolist()
                scores[idx-start_idx] = score
            for score in scores:
                self.out_file.write(score + '\n')
        return

    def translate_batch(self, batch, src_vocabs, attn_debug):
        """Translate a batch of sentences."""
        with torch.no_grad():
            if False:
                return self._translate_random_sampling(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    sampling_temp=self.random_sampling_temp,
                    keep_topk=self.sample_from_topk,
                    return_attention=attn_debug or self.replace_unk)
            else:
                return self._translate_batch(
                    batch,
                    src_vocabs,
                    self.max_length,
                    min_length=self.min_length,
                    ratio=self.ratio,
                    n_best=self.n_best,
                    return_attention=attn_debug or self.replace_unk)

    def _run_encoder(self, batch):
        src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                           else (batch.src, None)

        out = self.model.encoder(src, src_lengths)
        return out

        enc_states, memory_bank, src_lengths = self.model.encoder(
            src, src_lengths)
        if src_lengths is None:
            assert not isinstance(memory_bank, tuple), \
                'Ensemble decoding only supported for text data'
            src_lengths = torch.Tensor(batch.batch_size) \
                               .type_as(memory_bank) \
                               .long() \
                               .fill_(memory_bank.size(0))
        return src, enc_states, memory_bank, src_lengths

    def _translate_batch(
            self,
            batch,
            src_vocabs,
            max_length,
            min_length=0,
            ratio=0.,
            n_best=1,
            return_attention=False):
        # TODO: support these blacklisted features.
        assert not self.dump_beam

        outputs = self._run_encoder(batch)
        return outputs
        # (0) Prep the components of the search.
        use_src_map = self.copy_attn
        beam_size = self.beam_size
        batch_size = batch.batch_size

        # (1) Run the encoder on the src.
        src, enc_states, memory_bank, src_lengths = self._run_encoder(batch)
        self.model.decoder.init_state(src, memory_bank, enc_states)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            "batch": batch,
            "gold_score": self._gold_score(
                batch, memory_bank, src_lengths, src_vocabs, use_src_map,
                enc_states, batch_size, src)}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        src_map = (tile(batch.src_map, beam_size, dim=1)
                   if use_src_map else None)
        self.model.decoder.map_state(
            lambda state, dim: tile(state, beam_size, dim=dim))

        if isinstance(memory_bank, tuple):
            memory_bank = tuple(tile(x, beam_size, dim=1) for x in memory_bank)
            mb_device = memory_bank[0].device
        else:
            memory_bank = tile(memory_bank, beam_size, dim=1)
            mb_device = memory_bank.device
        memory_lengths = tile(src_lengths, beam_size)

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            n_best=n_best,
            batch_size=batch_size,
            global_scorer=self.global_scorer,
            pad=self._tgt_pad_idx,
            eos=self._tgt_eos_idx,
            bos=self._tgt_bos_idx,
            min_length=min_length,
            ratio=ratio,
            max_length=max_length,
            mb_device=mb_device,
            return_attention=return_attention,
            stepwise_penalty=self.stepwise_penalty,
            block_ngram_repeat=self.block_ngram_repeat,
            exclusion_tokens=self._exclusion_idxs,
            memory_lengths=memory_lengths)

        for step in range(max_length):
            decoder_input = beam.current_predictions.view(1, -1, 1)

            log_probs, attn = self._decode_and_generate(
                decoder_input,
                memory_bank,
                batch,
                src_vocabs,
                memory_lengths=memory_lengths,
                src_map=src_map,
                step=step,
                batch_offset=beam._batch_offset)

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                if isinstance(memory_bank, tuple):
                    memory_bank = tuple(x.index_select(1, select_indices)
                                        for x in memory_bank)
                else:
                    memory_bank = memory_bank.index_select(1, select_indices)

                memory_lengths = memory_lengths.index_select(0, select_indices)

                if src_map is not None:
                    src_map = src_map.index_select(1, select_indices)

            self.model.decoder.map_state(
                lambda state, dim: state.index_select(dim, select_indices))

        results["scores"] = beam.scores
        results["predictions"] = beam.predictions
        results["attention"] = beam.attention
        return results

    def _report_score(self, name, score_total, words_total):
        if words_total == 0:
            msg = "%s No words predicted" % (name,)
        else:
            msg = ("%s AVG SCORE: %.4f, %s PPL: %.4f" % (
                name, score_total / words_total,
                name, math.exp(-score_total / words_total)))
        return msg

    def _report_bleu(self, tgt_path):
        import subprocess
        base_dir = os.path.abspath(__file__ + "/../../..")
        # Rollback pointer to the beginning.
        self.out_file.seek(0)
        print()

        res = subprocess.check_output(
            "perl %s/tools/multi-bleu.perl %s" % (base_dir, tgt_path),
            stdin=self.out_file, shell=True
        ).decode("utf-8")

        msg = ">> " + res.strip()
        return msg

    def _report_rouge(self, tgt_path):
        import subprocess
        path = os.path.split(os.path.realpath(__file__))[0]
        msg = subprocess.check_output(
            "python %s/tools/test_rouge.py -r %s -c STDIN" % (path, tgt_path),
            shell=True, stdin=self.out_file
        ).decode("utf-8").strip()
        return msg
