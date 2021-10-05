from PyPDF2 import PdfFileReader
import matplotlib.pyplot as plt
from utils.ocr_utils import ImageProfile, columnDetectionLPF
import pickle
from utils.crf_utils import ContionalRandomField

resolution_single = 211  # single column -> 211 dpi
resolution_double = 255  # double column -> 255 dpi

pdf_folder = 'E:/task3/dataset/'
filenames = [str(i) for i in range(1, 103)]

with open('saved_models/crf.pkl', 'rb') as f:
    crf = pickle.load(f)

for filename in filenames:
    print('Processing ' + filename + ' ...')
    # with open('E:\zelun\output_bleu_model/' + filename + '.txt',
    #           'r', encoding="utf8") as file:
    #     text = file.read().replace('\n', ' ')
    pdf_path = pdf_folder + filename + '.pdf'
    # pdf = PdfFileReader(open(pdf_path, 'rb'))
    # p = pdf.getPage(0)
    # pdf_h = p.mediaBox.getHeight()

    pil_img = ImageProfile.PDF2PIL(pdf_path, resolution_single)
    # width, height = pil_img.size
    # # ratio of img_size/pdf_size
    # ratio = height / pdf_h
    images = columnDetectionLPF(pil_img)

    # if the page has two columns, use a different resolution
    if len(images) == 2:
        pil_img = ImageProfile.PDF2PIL(pdf_path, resolution_double)
        images = columnDetectionLPF(pil_img)

    text = ''
    for image in images:
        # create the image profile
        columnProfile = ImageProfile(image)
        # columnProfile.drawXProjection()
        # columnProfile.drawLines()
        columnProfile.runOCR(pdf_folder, filename)
        # columnProfile.drawWords()
        # with open(filename + '.pkl', 'wb') as f:
        #     pickle.dump(columnProfile, f)

        crf.predict(columnProfile)

        # merge math, detect IME
        columnProfile.postProcessing()

        columnProfile.translateLaTeX()

        text += columnProfile.outputText()

    text = ImageProfile.spellcheck(text)

    f = open('output_bleu_model_spellchecker/' + filename + '.txt', "w")
    f.write(text)
    f.close()
