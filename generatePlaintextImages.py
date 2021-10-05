# deprecated
# this script generates data from Marmot dataset processed by pdf parser-based algorithm

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
from utils.image_utils import downsample_image, pad_group_image


pdf_folder = 'E:/zelun/imgs2/'

image_dir = pdf_folder + '/images'
if save_images and not os.path.exists(image_dir):
    os.mkdir(image_dir)

files = os.listdir(pdf_folder)

text_file = open(pdf_folder + 'src.txt', "w")

num = 0
for cnt in range(5):
    # for pn in range(1,103):
    for pn in files:
        if pn[-3:] != 'pdf':
            continue
        num += 1
        pn = pn[:-4]
        pdf_path = pdf_folder + str(pn) + '.pdf'
        box_path = pdf_folder + str(pn) + '.pkl'

        # width and height of the original pdf
        pdf = PdfFileReader(open(pdf_path, 'rb'))
        p = pdf.getPage(0)
        h = p.mediaBox.getHeight()
        # convert to image
        print("converting PDF to image...")
        try:
            with(wandimage(filename=pdf_path, resolution=random.randint(200,255))) as source:
                source.alpha_channel = False
                img_buffer = np.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
                bytesio = io.BytesIO(img_buffer)
                pil_img = Image.open(bytesio)
                pil_img.save(os.path.splitext(pdf_path)[0] + '.png')
                # ratio of img_size/pdf_size
                ratio = pil_img.size[1] / h
            print("crop image to MEs...")
        except:
            continue


        with open(box_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            box_data = u.load()

        old_im = pil_img.convert('L')
        # draw = ImageDraw.Draw(old_im)
        for idx, box in enumerate(box_data['me_boxes']):
            box = [v*float(ratio) for v in box]
            new_im = old_im.crop((box[0]-1, pil_img.size[1]-box[3]-1,
                                 box[2]+1, pil_img.size[1]-box[1]+1))
            # draw.rectangle([box[0]-1, pil_img.size[1]-box[1]+1, box[2]+1, pil_img.size[1]-box[3]-1], outline="#FF0000")
            try:
                new_im = downsample_image(new_im, ratio)
            except:
                continue
            new_im = pad_group_image(new_im, padding_size, buckets)

            image_path = pdf_folder + '\images\\' + '' + hex(num+cnt*102)[2:] + '_' + hex(idx)[2:] + '.png'
            new_im.save(image_path)
            text_file.write('' + hex(num+cnt*102)[2:] + '_' + hex(idx)[2:] + '.png' + "\n")
text_file.close()

