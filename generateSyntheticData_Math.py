# this script generates synthetic data for math
# maths.text is a file, with each line containing a LaTeX source for math

import os
from utils.consts import save_images
from wand.image import Image as wandimage
import random
import numpy as np
import io
from PIL import Image
from utils.image_utils import latex_to_image, crop_image


pdf_folder = 'E:/zelun/synthetic/'

image_dir = pdf_folder + '/maths'
if save_images and not os.path.exists(image_dir):
    os.mkdir(image_dir)

maths = open(pdf_folder + 'maths.txt').readlines()

for idx, math in enumerate(maths):
    filename = 'm_' + hex(idx)[2:]
    try:
        latex_to_image(math, image_dir + '/' + filename, image_dir)
        pdf_path = image_dir + '/' + filename + '.pdf'
        # convert to image
        with(wandimage(filename=pdf_path, resolution=random.randint(180,255))) as source:
            source.alpha_channel = False
            img_buffer = np.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
            bytesio = io.BytesIO(img_buffer)
            pil_img = Image.open(bytesio)

            pil_img = crop_image(pil_img)

            # pil_img = downsample_image(pil_img, 2.0)
            # pil_img = pad_group_image(pil_img, padding_size, buckets)

            pil_img.save(os.path.splitext(pdf_path)[0] + '.png')
            os.remove(pdf_path)
    except:
        continue
