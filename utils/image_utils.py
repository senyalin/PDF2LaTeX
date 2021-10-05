from PIL import Image
from wand.image import Image as wandimage
import numpy as np
from PyPDF2 import PdfFileReader
import xml.etree.ElementTree as ET
from utils.consts import padding_size, buckets, downsample_ratio,\
    image_resolution, latex_template, TIMEOUT, tmp_folder, save_images,\
    latex_template_plaintext, latex_template_bf, latex_template_it,\
    latex_template_sec, latex_template_para, latex_template_bfif
import io
import re
import os
import subprocess
from threading import Timer
import random


def convert_pdf_to_images(pdf_path, column_number):
    pdf_name = os.path.basename(pdf_path)[:-4]
    me_box_path = os.path.dirname(os.path.abspath(pdf_path)) + '/' + pdf_name + '.xml'
    # width and height of the original pdf
    pdf = PdfFileReader(open(pdf_path, 'rb'))
    p = pdf.getPage(0)
    h = p.mediaBox.getHeight()
    if column_number == '1':
        resolution = 211
    else:
        resolution = 255
    # convert to image
    print("converting PDF to image...")
    with(wandimage(filename=pdf_path, resolution=resolution)) as source:
        source.alpha_channel = False
        img_buffer = np.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
        bytesio = io.BytesIO(img_buffer)
        pil_img = Image.open(bytesio)
        pil_img.save(os.path.splitext(pdf_path)[0] + '.png')
        # ratio of img_size/pdf_size
        ratio = pil_img.size[1] / h
    print("crop image to MEs...")
    crop_pdf(pdf_path, pil_img, me_box_path, ratio)


def crop_pdf(pdf_path, img, xml_path, ratio):
    if not os.path.exists(tmp_folder):
        os.mkdir(tmp_folder)
    text_file = open(tmp_folder + 'src.txt', "w")
    old_im = img.convert('L')
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # draw = ImageDraw.Draw(old_im)
    for i, child in enumerate(root):
        if child[0].tag != 'LineBox':
            bbox = child.attrib['BBox']
            box = [float(v) * float(ratio) for v in bbox.split()]
            new_im = old_im.crop((box[0]-1, old_im.size[1]-box[3]-1,
                                  box[2]+1, old_im.size[1]-box[1]+1))
            new_im = downsample_image(new_im, downsample_ratio)
            new_im = pad_group_image(new_im, padding_size, buckets)
            # new_im = downsample_image(new_im, downsample_ratio)
            image_path = os.path.dirname(pdf_path) + "/images/" + \
                         os.path.basename(pdf_path)[:-4] + '_' + str(i) + ".png"
            if save_images:
                new_im.save(image_path)
            text_file.write(image_path + "\n")
        else:
            # this is to handle multi-line IME
            idx = 0
            while child[idx].tag == 'LineBox':
                bbox = child[idx].attrib['Box']
                box = [float(v) * float(ratio) for v in bbox.split()]
                new_im = old_im.crop((box[0] - 1, old_im.size[1] - box[3] - 1,
                                      box[2] + 1, old_im.size[1] - box[1] + 1))
                new_im = downsample_image(new_im, downsample_ratio)
                new_im = pad_group_image(new_im, padding_size, buckets)
                # new_im = downsample_image(new_im, downsample_ratio)
                image_path = os.path.dirname(pdf_path) + "/images/" + os.path.basename(pdf_path)[:-4]\
                             + '_' + str(i) + '.' + str(idx) + ".png"
                if save_images:
                    new_im.save(image_path)
                text_file.write(image_path + "\n")
                idx += 1
    text_file.close()


def pad_group_image(old_im, pad_size, buckets):
    PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = pad_size
    old_size = (old_im.size[0]+PAD_LEFT+PAD_RIGHT, old_im.size[1]+PAD_TOP+PAD_BOTTOM)
    j = -1
    for i in range(len(buckets)):
        if old_size[0]<=buckets[i][0] and old_size[1]<=buckets[i][1]:
            j = i
            break
    if j < 0:
        new_size = old_size
        new_im = Image.new("RGB", new_size, (255,255,255))
        new_im.paste(old_im, (PAD_LEFT,PAD_TOP))
        return new_im
    new_size = buckets[j]
    new_size = random.choice([[120,64],[120,64],[120,64],[120,64],[120,64],[120,64],[120,64],
                              [120,160],[120,160],[120,160],[120,160],[120,160],
                              [240,64],[240,64],[240,64],[240,64],
                              [240,160],[240,160],[240,160],[240,160],
                              [360,64],[360,64],[360,64],[360,64],
                              [360, 160],[360,160],[360, 160],
                              [480, 64], [480, 100], [480, 160],
                              [480, 64], [480, 100], [480, 160],
                              [480,200],[660,64],[480,200],[660,64],
                              [660,100],[660,160],[800,100]])
    new_size = old_size
    new_im = Image.new("RGB", new_size, (255,255,255))
    new_im.paste(old_im, (PAD_LEFT,PAD_TOP))
    return new_im


def downsample_image(img, ratio):
    assert ratio>=1, ratio
    if ratio == 1:
        return img
    old_im = img
    old_size = old_im.size
    new_size = (int(old_size[0]/ratio), int(old_size[1]/ratio))
    new_im = old_im.resize(new_size, Image.LANCZOS)
    return new_im


def crop_image(img, default_size=None):
    old_im = img.convert('L')
    img_data = np.asarray(old_im, dtype=np.uint8) # height, width
    nnz_inds = np.where(img_data!=255)
    if len(nnz_inds[0]) == 0:
        if not default_size:
            return old_im
        else:
            assert len(default_size) == 2, default_size
            x_min,y_min,x_max,y_max = 0,0,default_size[0],default_size[1]
            old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
            return old_im
    y_min = np.min(nnz_inds[0])
    y_max = np.max(nnz_inds[0])
    x_min = np.min(nnz_inds[1])
    x_max = np.max(nnz_inds[1])
    old_im = old_im.crop((x_min, y_min, x_max+1, y_max+1))
    return old_im


# def crop_pdf_old(pdf_path, xml_path):
#     if os.path.isfile(os.path.dirname(pdf_path) + "/0.pdf"):
#         print('pdf already cropped!')
#         return
#     with open(pdf_path, "rb") as pdf_file:
#         input = PdfFileReader(pdf_file)
#         tree = ET.parse(xml_path)
#         root = tree.getroot()
#         for i, child in enumerate(root):
#             bbox = child.attrib['readable_bbox']
#             box = [float(v) for v in bbox.split()]
#             output = PdfFileWriter()
#             page = input.getPage(0)
#             page.trimBox.lowerLeft = (box[0], box[3])
#             page.trimBox.upperRight = (box[2], box[1])
#             page.cropBox.lowerLeft = (box[0], box[3])
#             page.cropBox.upperRight = (box[2], box[1])
#             output.addPage(page)
#             with open(os.path.dirname(pdf_path) + '/' + str(i) + ".pdf", "wb") as out_f:
#                 output.write(out_f)

def remove_space_in_latex(formula):
    # remove the spaces
    tokens = formula.split(' ')
    tokens = [t for t in tokens if t]
    latex = ''
    for t, tok in enumerate(tokens):
        if '\\' == tok[0]:
            tok += " "
        latex += tok
    # for i in range(len(latex)):
    #     if latex[i] == '\n':
    #         break
    # return latex[:i]
    return latex

def read_formulas(formula_path, idx_path):
    # read formulas and merge multi-line split
    formulas = open(formula_path).readlines()
    indexes = open(idx_path).readlines()
    tmp = ''
    ans = []
    for i in range(len(formulas)):
        path = indexes[i].split('_')
        path = path[-1][:-5]
        if formulas[i][-1] == '\n':
            formulas[i] = formulas[i][:-1]
        if '.' not in path:
            if tmp:
                ans.append(tmp[2:])
                tmp = ''
            ans.append(formulas[i])
        else:
            if int(path.split('.')[0]) != len(ans):
                ans.append(tmp[2:])
                tmp = ''
            tmp += '\\\\' + formulas[i]
    if tmp:
        ans.append(tmp[2:])
    return ans

def generate_latex(pdf_path):
    input_txt = os.path.splitext(pdf_path)[0] + '.txt'
    formula_path = os.path.splitext(pdf_path)[0] + '_pred.txt'
    idx_path = os.path.splitext(pdf_path)[0] + '_src.txt'
    output_path = os.path.splitext(pdf_path)[0] + '_latex.txt'
    formulas = read_formulas(formula_path, idx_path)

    # for i, formula in enumerate(formulas):
    #     name_prefix = output_path + '/' + os.path.basename(pdf_path)[:-4] + '_' + str(i)
    #     latex_to_image(formula, name_prefix, output_path)
    me_idx = 0
    fout = open(output_path, "w", encoding="utf-8")
    with open(input_txt, encoding="utf-8") as fin:
        for line in fin:
            line = line.split(' ')
            for word in line:
                if re.match("\$me_id_\d+_me_id\$", word):
                    word = '$' + remove_space_in_latex(formulas[me_idx]) + '$'
                    me_idx += 1
                fout.write(word)
                if word != '\n':
                    fout.write(' ')
            fout.write('\\\\')
    fout.close()


def batch_latex_to_image(pdf_path):
    if not save_images:
        return
    input_path = os.path.dirname(pdf_path) + '/'
    output_path = input_path + 'images/'
    label_path = input_path + os.path.basename(pdf_path)[:-4] + '_pred.txt'
    idx_path = input_path + os.path.basename(pdf_path)[:-4] + '_src.txt'
    indexes = open(idx_path).readlines()
    formulas = open(label_path).readlines()
    for i, formula in enumerate(formulas):
        path = indexes[i].split('_')
        path = path[-1][:-5]
        name_prefix = output_path + '/' + os.path.basename(pdf_path)[:-4] + '_' + path
        latex_to_image(formula, name_prefix, output_path)


def latex_to_image(formula, name_prefix, output_path):
    formula = formula.strip()
    formula = formula.replace(r'\pmatrix', r'\mypmatrix')
    formula = formula.replace(r'\matrix', r'\mymatrix')
    formula = formula.strip('%')
    if len(formula) == 0:
        formula = '\\hspace{1cm}'
        for space in ["hspace", "vspace"]:
            match = re.finditer(space + " {(.*?)}", formula)
            if match:
                new_l = ""
                last = 0
                for m in match:
                    new_l = new_l + formula[last:m.start(1)] + m.group(1).replace(" ", "")
                    last = m.end(1)
                new_l = new_l + formula[last:]
                formula = new_l
    tex_filename = name_prefix + '.tex'
    with open(tex_filename, "w") as w:
        print((latex_template % formula), file=w)
    run_pdflatex("C:/texlive/2018/bin/win32/pdflatex %s" % tex_filename, output_path, TIMEOUT)
    os.remove(tex_filename)
    os.remove(name_prefix + '.log')
    os.remove(name_prefix + '.aux')

def latex_plaintext_to_image(text, name_prefix, output_path):
    tex_filename = name_prefix + '.tex'
    rnd = random.randint(1, 100)
    # choose random font
    if rnd == 1:
        template = latex_template_bf
    elif rnd == 2:
        template = latex_template_it
    elif rnd == 3:
        template = latex_template_sec
    elif rnd == 4:
        template = latex_template_para
    elif rnd == 5:
        template = latex_template_bfif
    else:
        template = latex_template_plaintext
    with open(tex_filename, "w") as w:
        print((template % text), file=w)
    run_pdflatex("C:/texlive/2018/bin/win32/pdflatex %s" % tex_filename, output_path, TIMEOUT)
    os.remove(tex_filename)
    os.remove(name_prefix + '.log')
    os.remove(name_prefix + '.aux')

def run_pdflatex(cmd, output_path, timeout_sec):
    proc = subprocess.Popen(cmd, cwd=output_path)
    kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()
