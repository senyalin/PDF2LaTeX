from pdfminer.layout import LTChar
from utils.box_utils import get_char_list_bbox
import numpy as np
from utils.consts import chunk_exception_list, concatenation_operators, bullet_list, \
    math_words, short_math_words, left_parentheses, right_parentheses, script_ratio
from utils.cons_seq_prediction import pairwise_model
import re
from utils.box_utils import is_bind_var_line, vdist_between_two_lines
from utils.rules import valid_parenthesis_rule, merge_spatial_chunks_rule, split_symbols_rule, \
    IME_regex_right, IME_regex_left,is_line_ime_rule, theorem_regex, section_regex

# load word library for plain-text checking
from nltk.corpus import words
from nltk.stem.wordnet import WordNetLemmatizer
from utils.consts import additional_words
wl = set(words.words())
wl.update(additional_words)
wnl = WordNetLemmatizer()


def separate_equa_num(ime_chunk_list):
    # separate IME index from the IME body
    if re.match(IME_regex_right, ime_chunk_list['text']):
        for i in reversed(range(len(ime_chunk_list['symbols']))):
            if ime_chunk_list['symbols'][i].raw_text == '(':
                idx_chunk = ime_chunk_list['symbols'][i:]
                ime_chunk = ime_chunk_list['symbols'][:i]
                break
    elif re.match(IME_regex_left, ime_chunk_list['text']):
        for i in range(len(ime_chunk_list['symbols'])):
            if ime_chunk_list['symbols'][i].raw_text == ')':
                idx_chunk = ime_chunk_list['symbols'][:i+1]
                ime_chunk = ime_chunk_list['symbols'][i+1:]
                break
    else:
        idx_chunk = []
        ime_chunk = ime_chunk_list['symbols']
    idx = []
    for c in idx_chunk:
        idx.append(c.get_text())

    return ime_chunk, ''.join(idx)


def merge_splitted_imes(chunk_list_list, hist_info, left_bound):
    new_chunk_list_list = []
    merge_pool = []
    for i, chunk_list in enumerate(chunk_list_list):
        merge_pool.append(chunk_list)
        if i < len(chunk_list_list) - 1 and \
                is_line_ime_rule(chunk_list_list[i], hist_info, left_bound) and \
                is_line_ime_rule(chunk_list_list[i+1], hist_info, left_bound) \
                and (chunk_list_list[i][0]['symbols'][-1].get_text() in concatenation_operators
                     or chunk_list_list[i][0]['symbols'][-1].raw_text in concatenation_operators
                     or chunk_list_list[i + 1][0]['symbols'][0].get_text() in concatenation_operators
                     or chunk_list_list[i + 1][0]['symbols'][0].raw_text in concatenation_operators)\
                and not re.match('^.*\(\d+(.\d+)?\)$', chunk_list_list[i][0]['text']):
            continue
        if len(merge_pool) == 1:
            new_chunk_list_list.append(merge_pool[0])
        else:
            chunk_pool = []
            ime_bbox = []
            for item in merge_pool:
                chunk_pool.extend(item)
                ime_bbox.append(item[0]['bbox'])
            new_chunk = merge_chunks(chunk_pool)
            new_chunk['ime_bbox'] = ime_bbox
            new_chunk_list_list.append([new_chunk])
        merge_pool = []
    if merge_pool:
        if len(merge_pool) == 1:
            new_chunk_list_list.append(merge_pool[0])
        else:
            chunk_pool = []
            for item in merge_pool:
                chunk_pool.extend(item)
            new_chunk_list_list.append([merge_chunks(chunk_pool)])

    return new_chunk_list_list


def merge_middle_symbols(lines, hist_info):
# if symbols in between MEs are pure numbers or ',...,'
# merge them into one ME
    new_lines = []
    # load word library for plain-text checking
    for line in lines:
        merge_pool = []
        new_line = []
        for i, chunk in enumerate(line):
            if chunk['me_candidate']:
                if merge_pool:
                    merged_chunk = merge_chunks(merge_pool)
                    if merged_chunk['text'].isdigit() or re.match('[;,](\.|·)+[;,]', merged_chunk['text']):
                        merged_chunk['me_candidate'] = True
                        new_line.append(merged_chunk)
                    else:
                        new_line.extend(merge_pool)
                new_line.append(chunk)
                merge_pool = []
            else:
                merge_pool.append(chunk)
        if merge_pool:
            merged_chunk = merge_chunks(merge_pool)
            if re.match('[;,]\.+[;,]', merged_chunk['text']) and \
                    merged_chunk['text'].isdigit() and len(line) > 1:
                merged_chunk['me_candidate'] = True
                new_line.append(merged_chunk)
            else:
                new_line.extend(merge_pool)
        new_lines.append(new_line)
    # merge consecutive me chunks
    chunk_list_list = []
    for li, line in enumerate(new_lines):
        new_line = []
        merge_list = []
        for chunk in line:
            if not chunk['me_candidate']:
                if merge_list:
                    merged_chunk = merge_chunks(merge_list)
                    if merged_chunk['me_candidate'] and \
                            is_false_detection(merged_chunk, hist_info['text_heights'][li]):
                        merged_chunk['me_candidate'] = False
                    new_line.append(merged_chunk)
                    merge_list = []
                new_line.append(chunk)
            else:
                merge_list.append(chunk)
        if merge_list:
            merged_chunk = merge_chunks(merge_list)
            if merged_chunk['me_candidate'] and \
                    is_false_detection(merged_chunk, hist_info['text_heights'][li]):
                merged_chunk['me_candidate'] = False
            new_line.append(merged_chunk)

        chunk_list_list.append(new_line)

    return chunk_list_list


def is_false_detection(chunk, text_height):
    # remove pure numbers
    if chunk['text'].isdigit() or chunk['text'] == ':':
        return True
    # remove plaintext words
    word = chunk['text']
    word = word.lower().strip()
    # print check word
    try:
        s_word = wnl.lemmatize(word, 'n')
        v_word = wnl.lemmatize(word, 'v')
    except:  # special symbols may fail to lemmatize
        s_word = ''
        v_word = ''
    if len(word) > 2 and word != 'null' and \
            (word in wl or s_word in wl or v_word in wl):
        return True
    # remove long text word

    # after this loop, we know all alphabetic symbols are main body text
    alpha_size = None
    for symbol in chunk['symbols']:
        if symbol.raw_text.isalpha():
            if not alpha_size:
                alpha_size = symbol.size
            if alpha_size != symbol.size:
                return False
    # start with a (super)script number indicates an annotation
    if chunk['text'][0].isdigit() and float(chunk['symbols'][0].size) < script_ratio * text_height:
        return True
    if re.match('^[a-z]*(:|(\*|∗))$', chunk['text'], re.IGNORECASE) or\
            re.match('^(\*|∗)[a-z]*$', chunk['text'], re.IGNORECASE):
        # e.g., 'methods:', '*definition'
        return True
    # if chunk['text'][0] ==
    # if chunk['text'] == 'methods:':
    #     return True
    return False

def split_commas(lines, hist_info, left_bound):
    # if comma is surrounded by two numbers, or one period, do not split. Otherwise, split by comma
    # split the split_symbols_rule at the end of MEs

    new_lines = []
    from utils.groupChunk import get_chunk_struct
    for li, line in enumerate(lines):
        new_line = []
        if not is_line_ime_rule(line, hist_info, left_bound):
            for chunk in line:
                # if a chunk in this stage is still a single word, it's probably not an ME
                if chunk['text'] in short_math_words:
                    chunk['me_candidate'] = True
                if not chunk['me_candidate'] or \
                        not any(s in chunk['text'] for s in split_symbols_rule):
                    new_line.append(chunk)
                else:
                    has_parenthesis = False
                    if any(parenthesis in chunk['text'] for parenthesis
                           in left_parentheses + right_parentheses):
                        has_parenthesis = True
                    new_chunk = []
                    for i, symbol in enumerate(chunk['symbols']):
                        # remove the ending periods, commas, etc.
                        if i == len(chunk['symbols'])-1:
                            if chunk['symbols'][i].raw_text in split_symbols_rule:
                                if new_chunk:
                                    new_line.append(get_chunk_struct(new_chunk))
                                    new_chunk = []
                            new_chunk.append(symbol)
                            continue
                        # split MEs by commas
                        if symbol.raw_text == ',' and i > 0 and (not has_parenthesis) and\
                                (not ((chunk['symbols'][i-1].raw_text.isdigit() and
                                chunk['symbols'][i+1].raw_text.isdigit()) or
                                (chunk['symbols'][i-1].raw_text in '·.' or
                                chunk['symbols'][i+1].raw_text in '·.'))) and \
                                float(symbol.size) / hist_info['text_heights'][li] > script_ratio:
                                # do not split if it's a sub/sup-script comma
                            if new_chunk:
                                new_line.append(get_chunk_struct(new_chunk))
                                new_chunk = []
                            symbol.me_candidate = False
                            new_line.append(get_chunk_struct([symbol]))
                        else:
                            new_chunk.append(symbol)
                    if new_chunk:
                        new_line.append(get_chunk_struct(new_chunk))
        else:
            new_line = line

        new_lines.append(new_line)

    return new_lines


def merge_close_chunks(chunk_list_list, thres):
    """
    similar to insert_space_using_fontbox, but on chunk level
    """
    new_chunk_list_list = []
    for chunk_list in chunk_list_list:
        # merge spatially close chunks
        new_chunk_list = merge_spatial_chunks_rule(chunk_list, thres)
        # merge unmatched parenthesis
        new_chunk_list = valid_parenthesis_rule(new_chunk_list)
        # merge spatially close chunks
        new_chunk_list = merge_spatial_chunks_rule(new_chunk_list, thres)

        new_chunk_list_list.append(new_chunk_list)

    return new_chunk_list_list


def get_ime_line_idx(lines, hist_info, left_bound):
    """
    With math symbol and without non-math words
    Return:
        The index of lines that are ME
    """
    # IME assessment core
    for li, line in enumerate(lines):
        with_math_symbol_or_word = False
        with_non_math_word = False
        with_large_spatial_gap = False
        for i, chunk in enumerate(line):
            if chunk['me_candidate']:
                with_math_symbol_or_word = True
                continue
            word = chunk['text']
            word = word.lower().strip()

            if word in math_words:
                with_math_symbol_or_word = True
            elif word == 'null':
                continue
            elif len(word) >= 2 and chunk['plain_text']:
                with_non_math_word = True
                break
            else:
                pass
        merged_chunk = merge_chunks(lines[li])
        if (with_math_symbol_or_word and not with_non_math_word and
            not with_large_spatial_gap) or end_with_parenthesis(merged_chunk, hist_info):
            # detect if this IME line is a section line
            if re.match(section_regex, merged_chunk['text']):
                merged_chunk['me_candidate'] = False
            lines[li] = [merged_chunk]

    new_lines = []
    for i, line in enumerate(lines):
        if i == 0 or i == len(lines)-1:
            check_if_title_line(line, i, hist_info)
        new_line = []
        if not is_line_ime_rule(line, hist_info, left_bound):
            for chunk in line:
                # square4, bullet are the item/bullet symbol.
                # Remove item symbols from ME candidates
                # multiply is one of the operators.
                # todo: add more operators into the list
                if chunk['symbol_num'] == 1 and chunk['me_candidate'] and\
                        (chunk['symbols'][0].get_text() in bullet_list or
                         chunk['symbols'][0].get_text() == 'multiply' or
                         chunk['symbols'][0].get_text() == 'minus' or
                         chunk['symbols'][0].get_text() == 'add'):
                    chunk['symbols'][0].me_candidate = False
                    chunk['me_candidate'] = False
                    new_line.append(chunk)
                # remove eme pure number
                # remove eme figures
                elif chunk['me_candidate'] and (is_float(chunk['text']) or
                    chunk['text'].lower().startswith('figure') or chunk['text'].lower().startswith('fig.') or
                    chunk['text'].lower().startswith('table') or is_author(chunk)):
                    chunk['me_candidate'] = False
                    new_line.append(chunk)
                # elif chunk['me_candidate'] and len(chunk['symbols']) >= 2 \
                #         and chunk['symbols'][0].raw_text == '(' and chunk['symbols'][-1].raw_text == ')':
                #     # remove EME outer parenthesis
                #     # this function is problematic for now because it removes the 'plain_text' attribute
                #     chunks = split_chunk_parenthesis(chunk)
                #     new_line.extend(chunks)
                else:
                    new_line.append(chunk)
        else:
            if is_float(line[0]['text']):
                # if pure number is detected as IME, set it to NME
                line[0]['me_candidate'] = False
            new_line = line
        new_lines.append(new_line)

    return new_lines


def is_author(chunk):
    if chunk['symbols'][0].get_text() == 'dagger' or\
        chunk['symbols'][-1].get_text() == 'dagger':
        return True

    return False


def merge_ime_bind_var(chunk_list_list, left_bound, hist_info, line_boxes):
    # we need this round because of bad encoding
    # index of lines with big operators
    big_operator_line_idx = []
    for i in range(len(chunk_list_list)):
        if is_line_ime_rule(chunk_list_list[i], hist_info, left_bound):
            big_operator_line_idx.append(i)
    # neighbors/bind_var candidates of big operator lines
    # a candidate neighbor line has to be bounded by a big operator line
    neighbors = {}
    # the dict key indicates that the key line is a neighbor
    # the dict val includes all candidates that it could bind to
    for big_op_line in big_operator_line_idx:
        prev_line = big_op_line - 1
        next_line = big_op_line + 1
        if prev_line >= 0 and \
                is_bind_var_line(prev_line, big_op_line, line_boxes, hist_info, None, chunk_list_list):
            if prev_line not in neighbors:
                neighbors[prev_line] = []
            neighbors[prev_line].append(big_op_line)
        if next_line < len(chunk_list_list) and \
                is_bind_var_line(next_line, big_op_line, line_boxes, hist_info, None, chunk_list_list) \
                and chunk_list_list[next_line][0]['text'][0] not in '<>=+-':
            # if the last line is a page number, do not merge
            if next_line == len(chunk_list_list)-1 and len(chunk_list_list[-1]) == 1 \
                    and chunk_list_list[-1][0]['text'].isdigit():
                chunk_list_list[-1][0]['me_candidate'] = False
                continue
            if next_line not in neighbors:
                neighbors[next_line] = []
            neighbors[next_line].append(big_op_line)

    # find closest big_op line to bind to
    merge_candidates = []
    for key, big_lines in neighbors.iteritems():
        min_dist = float('inf')
        for big_line in big_lines:
            curr_dist = vdist_between_two_lines(key, big_line, line_boxes)
            if curr_dist < min_dist:
                candidate_line = big_line
                min_dist = curr_dist
        # if a line is in candidates, it indicates that this line need to be merged with its next line
        merge_candidates.append(min(key, candidate_line))

    new_chunk_list_list = []
    merge_pool = []
    for i, char_list in enumerate(chunk_list_list):
        merge_pool.append(char_list)
        if i in merge_candidates:
            continue
        if len(merge_pool) == 1:
            new_chunk_list_list.append(merge_pool[0])
        else:
            chunk_pool = []
            for item in merge_pool:
                chunk_pool.extend(item)
            new_chunk_list_list.append([merge_chunks(chunk_pool)])
        merge_pool = []
    if merge_pool:
        if len(merge_pool) == 1:
            new_chunk_list_list.append(merge_pool[0])
        else:
            chunk_pool = []
            for item in merge_pool:
                chunk_pool.extend(item)
            new_chunk_list_list.append([merge_chunks(chunk_pool)])

    return new_chunk_list_list


def is_float(s):
    for c in s:
        if not(c.isdigit() or c == '.'
               or c == '(' or c == ')'):
            return False
    return True


def end_with_parenthesis(chunk, hist_info):
    if re.match(IME_regex_right, chunk['text']):
        for i in reversed(range(len(chunk['symbols']))):
            if chunk['symbols'][i].raw_text == '(':
                break
        if i != 0 and chunk['symbols'][i].bbox[0] -\
                chunk['symbols'][i-1].bbox[2] > hist_info['word_interval'] * 5:
            return True
    if re.match(IME_regex_left, chunk['text']):
        for i in range(len(chunk['symbols'])):
            if chunk['symbols'][i].raw_text == ')':
                break
        if i != len(chunk['symbols'])-1 and chunk['symbols'][i+1].bbox[0] - \
                chunk['symbols'][i].bbox[2] > hist_info['word_interval'] * 5:
            return True
    return False


def check_if_title_line(line, i, hist_info):
    # if a line is at the beginning or the end of a page,
    # and only contains numbers, periods, commas, and chars, parenthesis,
    # at the same time, the line gap is large than common
    # set this line to a title line (non-ME)
    # return
    title_symbols = ['.', ',', '(', ')', 'quoteright', 'dagger', 'asteriskmath']
    is_title = True
    for chunk in line:
        for char in chunk['symbols']:
            if not (char.raw_text in title_symbols or char.get_text() in title_symbols or
                    char.raw_text.isdigit() or char.raw_text.isalpha()):
                is_title = False

    line_gap = hist_info['spaces'][1] if i == 0 else hist_info['spaces'][-1]
    if line_gap < hist_info['common_space'] * 3 or\
            hist_info['heights'][i] > 2 * hist_info['common_height']:
        is_title = False

    if is_title:
        for chunk in line:
           chunk['me_candidate'] = False
           chunk['me_likelihood'] = 0


def chunk_var(chunk_list):
    variance = np.var([float(chunk.size) for chunk in chunk_list])
    if variance < 0.001:
        variance = 0  # prevent calculation error (almost zero)
    return variance


def chunk_likelihood(chunk_list):
    return np.max([chunk.me_likelihood for chunk in chunk_list])


def get_chunk_struct(char_list):
    # input is a list of LTChars (NSCS)
    # output is a struct with the chunk symbols (LTChars), bbox, me label, text
    # number of symbols (length), number of alphabetic symbols, alphabetic ratio

    chunk_info = {
        'symbols': char_list,
        'bbox': get_char_list_bbox(char_list),
        'me_candidate': False,
        'symbol_num': len(char_list),
        'var': chunk_var(char_list),  # variance of font size
        'alpha_num': 0,
        'alpha_ratio': 0,  # ratio of alphabetic symbols vs. non-alphabetic symbols
        'me_likelihood': chunk_likelihood(char_list)
    }

    # convert a chunk to a string
    text = []
    # calculate the alpha ratio
    for i in range(len(char_list)):
        text.append(char_list[i].raw_text)
        chunk_info['me_candidate'] = char_list[i].me_candidate or chunk_info['me_candidate']
        if char_list[i].raw_text.isalpha() and char_list[i].raw_text != 'null':
            chunk_info['alpha_num'] += 1
    chunk_info['text'] = ''.join(text)
    if chunk_info['text'] in chunk_exception_list:
        chunk_info['me_likelihood'] = 0
        chunk_info['me_candidate'] = False
    chunk_info['alpha_ratio'] = chunk_info['alpha_num'] * 1.0 / chunk_info['symbol_num']

    # use font variance and alpha_ratio info
    if chunk_info['var'] > 0:
        chunk_info['me_likelihood'] *= 2
    # if chunk_info['alpha_num'] == 0 and chunk_info['symbol_num'] > 1:
    #     chunk_info['me_likelihood'] *= 2
    chunk_info['me_likelihood'] = min(chunk_info['me_likelihood'], 1)

    return chunk_info


def split_chunk_parenthesis(chunk):
    chunk_list = []
    # the left parenthesis
    chunk_list.append(get_chunk_struct([chunk['symbols'][0]]))
    chunk_list[-1]['me_candidate'] = False
    # the middle math part excluding parenthesis
    math_part = chunk['symbols'][1:-1]
    if math_part:  # if non-empty
        chunk_list.append(get_chunk_struct(math_part))
        chunk_list[-1]['me_candidate'] = True
    # the right parenthesis
    chunk_list.append(get_chunk_struct([chunk['symbols'][-1]]))
    chunk_list[-1]['me_candidate'] = False
    return chunk_list


def merge_chunks(chunk_list):
    # input is a list chunk struct
    # output is one merged chunk struct
    char_list = []
    me_candidate = False
    for chunk in chunk_list:
        char_list.extend(chunk['symbols'])
        if chunk['me_candidate']:
            me_candidate = True
    char_list.sort(key=lambda char_arg: char_arg.bbox[0])
    merged_chunk = get_chunk_struct(char_list)
    merged_chunk['plain_text'] = False if len(chunk_list) != 1 \
        or 'plain_text' not in chunk_list[0] else chunk_list[0]['plain_text']
    merged_chunk['me_candidate'] = me_candidate
    return merged_chunk


def chars_to_chunks_using_me_labels(lines, hist_info):
    # converts lines of LTChars into chunks
    # using the information of me_candidate labels
    # different labels cannot be grouped into the same chunk
    # the label are from the symbol-level likelihood ratio test
    # This function also evaluates chunk-level ME likelihood

    chunk_list_list = []
    for li, line in enumerate(lines):
        if len(line) == 0:
            return []

        chunk = []
        chunk_list = []
        for c in line:
            if isinstance(c, LTChar):
                chunk.append(c)
            else:
                if len(chunk) > 0:
                    chunk_list.append(get_chunk_struct(chunk))
                    chunk = []

        if len(chunk) > 0:
            chunk_list.append(get_chunk_struct(chunk))

        # detect REFERENCE line
        if len(chunk_list) == 1 and \
                chunk_list[0]['text'] in ['REFERENCE', 'REFERENCES']\
                and hist_info['widths'][li] < hist_info['common_width']/2\
                and hist_info['spaces'][li] > hist_info['common_space']*2:
            break
        chunk_list_list.append(chunk_list)

    # set ME likelihood of all plain text to 0
    # set short math words according to short_math_words
    # todo: add a set of exception words (maximize, etc...)
    for line in chunk_list_list:
        for chunk in line:
            word = chunk['text']
            word = word.lower().strip()
            # print check word
            try:
                s_word = wnl.lemmatize(word, 'n')
                v_word = wnl.lemmatize(word, 'v')
            except: # special symbols may fail to lemmatize
                s_word = ''
                v_word = ''
            if len(word) > 1 and (word in wl or s_word in wl or v_word in wl):
                chunk['me_likelihood'] = 0
                chunk['plain_text'] = True
            else:
                chunk['plain_text'] = False
                if word in short_math_words:
                    chunk['me_likelihood'] = 0
                    chunk['me_candidate'] = True
    return chunk_list_list


def chunk_label_with_MRF(chunk_list_list):
    for li, line in enumerate(chunk_list_list):
        chunk_str_list = [chunk['text'] for chunk in line]
        # me_log_prob_list = [1 if chunk['me_candidate'] else 0 for chunk in line]
        # nme_log_prob_list = [0 if chunk['me_candidate'] else 1 for chunk in line]
        me_log_prob_list = [chunk['me_likelihood'] for chunk in line]
        nme_log_prob_list = [1-chunk['me_likelihood'] for chunk in line]
        nscs_label = pairwise_model({
            'me_log_prob_list': me_log_prob_list,
            'nme_log_prob_list': nme_log_prob_list,
            'nscs_str_list': chunk_str_list,
            'line_idx': li,
            'bin_weight': 1.0
        })
        for ci, chunk in enumerate(line):
            # preserve the me label from the earlier steps
            if not chunk['me_candidate'] and not chunk['plain_text']:
                chunk['me_candidate'] = nscs_label[ci] == 1
    return chunk_list_list


def is_chunk_candidate(chunk):
    math_phrases = {
        'min',
        'max',
        'lim',
        'minimize',
        'maximize',
        'mod'
    }
    ## do not consider commas and period as MEs so far
    # if chunk['text'] == '.' or chunk['text'] == ',':
    #     return False

    # consider non-ME chunks as ME chunks if alpha_ratio <= threshold
    # Todo: not using the threshold because some PDFs have bad encoding, which would mess up the alpha_ratio
    # threshold = 0.3 # -1 to filter out all non-MEs chunks
    # check the phrase dictionary
    if chunk['text'] in math_phrases:
        return True
    if chunk['me_candidate']:  # or chunk['alpha_ratio'] <= threshold:
        return True
    else:
        return False


def merge_consecutive_me_chunks(lines):
    # Todo: replace this method with more advanced merging methods, e.g., dilation
    # combines the consecutive ME candidate chunks
    # a chunk is classified as ME or NME based on the is_chunk_candidate() function

    chunk_list_list = []
    for line in lines:
        prev_len = len(line)
        while True:
            chunk_list = []
            merge_list = []
            temp_chunk = []
            for i, chunk in enumerate(line):
                # print chunk['text']
                if is_chunk_candidate(chunk):
                    # or (
                    #     i != 0 and is_chunk_candidate(line[i-1]) and
                    #     chunk['symbols'][0].get_text() in concatenation_operators):
                    merge_list.append(chunk)
                    if i != len(line)-1 and \
                            (chunk['symbols'][-1].get_text() in concatenation_operators or
                             chunk['symbols'][-1].raw_text in concatenation_operators):
                        line[i+1]['me_candidate'] = True
                    if len(merge_list) == 1 and i != 0 and \
                            (chunk['symbols'][0].get_text() in concatenation_operators or
                             chunk['symbols'][0].raw_text in concatenation_operators):
                        temp_chunk = chunk_list.pop(-1)
                        merge_list = [temp_chunk] + merge_list
                        temp_chunk = []
                else:
                    if merge_list:
                        # remove the leading and ending commas and periods
                        # remove the leading bullet points
                        if merge_list[0]['text'] == '.' or merge_list[0]['text'] == ',' \
                            or merge_list[0]['symbols'][0].get_text() in bullet_list:
                            temp_chunk = merge_list.pop(0)
                            chunk_list.append(temp_chunk)
                            temp_chunk = []
                        if merge_list and (merge_list[-1]['text'] == '.' or merge_list[-1]['text'] == ','):
                            temp_chunk = merge_list.pop(-1)
                        if merge_list:
                            merged_chunk = merge_chunks(merge_list)
                            if re.match(theorem_regex, merged_chunk['text'], re.IGNORECASE):
                                merged_chunk['me_candidate'] = False
                            chunk_list.append(merged_chunk)
                        merge_list = []
                        if temp_chunk:
                            chunk_list.append(temp_chunk)
                            temp_chunk = []
                    chunk_list.append(chunk)
            if merge_list:
                chunk_list.append(merge_chunks(merge_list))
            line = chunk_list
            if prev_len == len(line):
                break
            prev_len = len(line)
        chunk_list_list.append(line)

    return chunk_list_list

