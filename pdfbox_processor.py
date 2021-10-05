# -data_type img -model models/py-model.pt -src_dir pdfs/temp_data -src pdfs/temp_data/src.txt -output pdfs/temp_data/pred.txt -image_channel_size 3
# -data_type img -model models/BLEU_BEST_9028.pt -src_dir pdfs/temp_data -src pdfs/temp_data/src.txt -output pdfs/temp_data/pred.txt -image_channel_size 1

import translate
from utils.image_utils import convert_pdf_to_images, batch_latex_to_image, generate_latex
import os
import glob
from utils.consts import save_images
import shutil


pdf_folder = 'E:/task3/dataset/'
output_folder = 'E:\zelun\output_parser/'

filenames = [str(i) for i in range(78, 103)]

with open(pdf_folder + 'column_number.txt', 'r') as f:
    column_number = f.readlines()

for i, filename in enumerate(filenames):
    os.system("python .\pdf2me\me_extraction.py " + pdf_folder + " " + filename)

    image_dir = pdf_folder + filename + '/images'
    if save_images and not os.path.exists(image_dir):
        os.mkdir(image_dir)

    for pn in range(len(glob.glob(pdf_folder + filename + '/*.xml'))):
        pdf_path = pdf_folder + filename + '/' + filename + '_p' + str(pn) + '.pdf'

        convert_pdf_to_images(pdf_path, column_number[0][i])

        translate.run_external(pdf_path)

        generate_latex(pdf_path)

        # batch_latex_to_image(pdf_path)

        shutil.copyfile(pdf_folder + filename + '/' + filename + '_p0_latex.txt',
                        output_folder + filename + '.txt')

        # remove image folder
        try:
            shutil.rmtree(pdf_folder + filename)
        except:
            print('image deletion failure!')




# me_num = len([name for name in os.listdir(pdf_folder + pdf_name) if
#            name.endswith(".pdf") and name != pdf_name + '.pdf'])
# for i in range(me_num):
#     pdf_file = pdf_folder + pdf_name + '/' + str(i) + ".pdf"
#     output_file = os.path.splitext(pdf_file)[0] + '.png'
#     with(Image(filename=pdf_file, resolution=190)) as source:
#         # image = source.sequence[0]
#         # Image(image).save(filename=output_file)
#         # convert wand to PIL image
#         source.alpha_channel = False
#         img_buffer = numpy.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
#         bytesio = io.BytesIO(img_buffer)
#         pil_img = PIL.Image.open(bytesio)
#         pil_img = crop_image(pil_img)
#         pil_img = pad_group_image(pil_img, padding_size, buckets)
#         pil_img = downsample_image(pil_img, downsample_ratio)
#         pil_img.save(output_file)
