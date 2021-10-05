# this script generates synthetic data for plaintext
# each line in words.txt should contain one word

import os
from PyPDF2 import PdfFileReader
from utils.consts import save_images
from wand.image import Image as wandimage
import random
import numpy as np
import io
from PIL import Image#, ImageDraw
import pickle
from utils.consts import padding_size, buckets
from utils.image_utils import downsample_image, pad_group_image, latex_to_image,\
    latex_plaintext_to_image, crop_image


pdf_folder = 'E:/zelun/synthetic/'

image_dir = pdf_folder + '/ocr_images'
if save_images and not os.path.exists(image_dir):
    os.mkdir(image_dir)


# text_file = open(pdf_folder + 'words.txt', "r")
words = open(pdf_folder + 'words.txt').readlines()
fs = open("src.txt", "w")
ft = open("tgt.txt", "w")
num = 0
idx = 0
while True:
    if idx == len(words):
        break
    text = words[idx][:-1]
    idx += 1
    # if idx == len(words): break
    # rnd = random.randint(1, 200)
    # if rnd == 1:
    #     # 2-gram
    #     text += ' ' + words[idx][:-1]
    #     idx += 1
    #     if idx == len(words): break
    # elif rnd == 2:
    #     # 3-gram
    #     text += ' ' + words[idx][:-1]
    #     idx += 1
    #     text += ' ' + words[idx][:-1]
    #     idx += 1
    #     if idx == len(words): break
    filename = 'p_' + hex(idx)[2:]
    # if os.path.exists(image_dir + '/' + filename + '.png'):
    #     fs.write(filename + '.png' + '\n')
    #     ft.write(text + '\n')
    # else:
    #     continue
    # continue
    latex_plaintext_to_image(text, image_dir + '/' + filename, image_dir)
    pdf_path = image_dir + '/' + filename + '.pdf'
    # convert to image
    try:
        with(wandimage(filename=pdf_path, resolution=random.randint(180,255))) as source:
            source.alpha_channel = False
            img_buffer = np.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
            bytesio = io.BytesIO(img_buffer)
            pil_img = Image.open(bytesio)

            pil_img = crop_image(pil_img)
            # pil_img = downsample_image(pil_img, 2.0)
            pil_img = pad_group_image(pil_img, padding_size, buckets)

            pil_img.save(os.path.splitext(pdf_path)[0] + '.png')
            os.remove(pdf_path)

            fs.write(filename+'.png' + '\n')
            ft.write(text + '\n')

    except:
        continue
fs.close()
ft.close()