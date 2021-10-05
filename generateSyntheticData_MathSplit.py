# this script generates synthetic data for math, processed by profile projection cutting

import os
from PyPDF2 import PdfFileReader
from utils.consts import save_images
from PIL import Image#, ImageDraw
import pickle
from utils.consts import padding_size, buckets
from utils.image_utils import downsample_image, pad_group_image, latex_to_image, crop_image
from utils.ocr_utils import ImageProfile


pdf_folder = 'E:/zelun/synthetic/'

input_dir = pdf_folder + '/maths'
image_dir = pdf_folder + '/maths_split'

if save_images and not os.path.exists(image_dir):
    os.mkdir(image_dir)

files = os.listdir(input_dir)

for file in files:
    if file[-1] != 'g':
        continue
    print(file)
    img = Image.open(input_dir + '/' + file)

    # white padding
    # new_im = Image.new("RGB", (img.size[0] + 2,
    #         img.size[1] + 2), (255, 255, 255))
    new_im = img.convert('L')
    # new_im.paste(img, (1, 1))

    # new_im.show()

    imageProfile = ImageProfile(new_im, processMath=True)

    for idx, pil_img in enumerate(imageProfile.imgs):
        try:
            pil_img = downsample_image(pil_img, 2.0)
            pil_img = pad_group_image(pil_img, padding_size, buckets)
            pil_img.save(image_dir + '/' +
                         os.path.splitext(file)[0] + '_' + str(idx) + '.png')
        except:
            continue
