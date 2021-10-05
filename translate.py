#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from shutil import copyfile
import os
from utils.consts import tmp_folder, project_path


def run_external(pdf_path):
    print("Translating to Latex...")
    # parser = _get_parser()
    # opt = parser.parse_args()
    save_path = os.path.dirname(pdf_path) + '/'
    # main(opt, save_path)
    os.system('"C:/Users/Zelun Wang\AppData\Local\conda\conda\envs/v3\python.exe" ' + project_path + \
              'NeuralTranslator/translate.py ' + "-data_type img -model " + project_path + "saved_models/py-model.pt " + \
              "-src " + tmp_folder + "src.txt -output " + tmp_folder + "pred.txt " + \
              "-max_length 250 -image_channel_size 3")
    copyfile(tmp_folder + 'src.txt', save_path + os.path.basename(pdf_path)[:-4] + '_src.txt')
    copyfile(tmp_folder + 'pred.txt', save_path + os.path.basename(pdf_path)[:-4] + '_pred.txt')


def run_binaryCNN_external(pdf_path):
    print("Classifying ME labels...")
    save_path = os.path.dirname(pdf_path) + '/'
    os.system('"C:/Users/Zelun Wang\AppData\Local\conda\conda\envs/v3\python.exe" ' + project_path +\
        'BinaryCNN/translate.py ' + "-data_type img -model " + project_path + "saved_models/binary_20000.pt "+\
        "-src " + tmp_folder + "src.txt -output " + tmp_folder + "pred.txt " +\
        "-max_length 3 -beam_size 1 -batch_size 16 -image_channel_size 1")
    copyfile(tmp_folder + 'src.txt', save_path + os.path.basename(pdf_path)[:-4] + '_src.txt')
    copyfile(tmp_folder + 'pred.txt', save_path + os.path.basename(pdf_path)[:-4] + '_pred.txt')


def run_plaintextOCR_external(pdf_path):
    print("Plaintext OCR Translating...")
    save_path = os.path.dirname(pdf_path) + '/'
    os.system('"C:/Users/Zelun Wang\AppData\Local\conda\conda\envs/v3\python.exe" ' + project_path + \
              'NeuralTranslator/translate.py ' + "-data_type img -model " + project_path + "saved_models/plaintext_big_55000.pt " + \
              "-src " + tmp_folder + "src_ocr.txt -output " + tmp_folder + "pred_ocr.txt " + \
              "-max_length 150 -beam_size 2 -batch_size 16 -image_channel_size 1")
    copyfile(tmp_folder + 'pred_ocr.txt', save_path + os.path.basename(pdf_path)[:-4] + '_pred_ocr.txt')


def run_LaTeX_external():
    print("LaTeX Translating...")
    os.system('"C:/Users/Zelun Wang\AppData\Local\conda\conda\envs/v3\python.exe" ' + project_path + \
              'NeuralTranslator/translate.py ' + "-data_type img -model " + project_path + "saved_models/BLEU_BEST_9028.pt " + \
              "-src " + tmp_folder + "src.txt -output " + tmp_folder + "pred.txt " + \
              "-max_length 250 -image_channel_size 1")


# -data_type img -model models/py-model.pt -src_dir pdfs/temp_data -src pdfs/temp_data/src.txt -output pdfs/temp_data/pred.txt -image_channel_size 3

