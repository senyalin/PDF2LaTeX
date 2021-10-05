from collections import Counter
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import math
from wand.image import Image as wandimage
import io
from PIL import Image
import os
import translate
from utils.consts import downsample_ratio, tmp_folder, stopwords
from utils.image_utils import downsample_image, padding_size, remove_space_in_latex
import shutil
import nltk
from scipy import signal
from spellchecker import SpellChecker
import re


english_vocab = set(w.lower() for w in nltk.corpus.words.words())
# letter_vocab = set('bcdefghjklmnopqrstuvwxyz')
spell = SpellChecker(distance=1)  # set at initialization

class ImageProfile:
    def __init__(self, image, processMath=False):
        self.img = image
        # y_projection is used for line detection
        self.y_projection = self.calculateProjection(self.img, 'y')
        # detect lines based on the y_projection
        self.lines = self.parseLines(self.y_projection)
        # if input is math, process it as one line and split on x
        if processMath:
            # this is used for generating synthetic data
            if not self.lines:
                self.imgs = []
                return
            # calculate XProjection for the line
            line = self.lines[0]
            self.calculateLineXProj(line)
            line.LineToWords()
            imgs = []
            for word in line.words:
                # calculate y projection of a word
                img = self.img.crop((word.left, word.top, word.right, word.bottom))
                word.y_projection = self.calculateProjection(img, 'y')
                word.getTrueHeight()
                img = self.img.crop((word.left, word.top, word.right, word.bottom))
                imgs.append(img)
            self.imgs = imgs
            return
        # get the most common line height, line width, and line gap
        self.stat = self.getStats()
        # refine the line detection by splitting those that overlapped a bit
        self.splitLines()
        # calculate XProjection for each line
        for i, line in enumerate(self.lines):
            self.calculateLineXProj(line)
        # shrink the white spaces to calculate the true width of each line
        self.shrinkLineWidth()
        # refine the stats
        self.stat = self.getStats()
        # merge the small symbols according to distance
        self.mergeSmallLines()
        # split lines to words
        self.constructWords()
        self.stat = self.getStats()

    def calculateLineXProj(self, line):
        # calculate x projection of a line
        img = self.img.crop((0, line.top, self.img.size[0], line.bottom))
        line.x_projection = self.calculateProjection(img, 'x')

    def getStats(self):
        # get statistics such as baseline width, height and baseline gap
        # gap means the common height of a line-to-line gap
        heights = []
        gaps = []
        widths = []
        areas = []
        for i, line in enumerate(self.lines):
            heights.append(line.height)
            widths.append(line.width)
            areas.append(line.black_area)
            if i > 0:
                gaps.append(self.lines[i].top - self.lines[i-1].bottom)
        stat = dict()
        stat['baseHeight'] = Counter(heights).most_common(1)[0][0]
        stat['baseWidth'] = Counter(widths).most_common(1)[0][0]
        stat['baseLineGap'] = int(np.median(gaps))
        stat['baseArea'] = int(np.median(areas))
        return stat

    def parseLines(self, projection):
        # scan through the projection profile and extract all non-zero y_projections
        # and store each of them as a Line object
        lines = []
        y_projection = []
        start_pos = None
        black_area = 0
        for i in range(projection.size):
            if projection[i] > 0:
                if start_pos == None:
                    start_pos = i  # pixel is 0-based index
                y_projection.append(projection[i])
                black_area += projection[i]
            else:
                if start_pos == None:
                    continue  # current y_projection is empty
                line = Line(
                    start_pos,
                    start_pos + len(y_projection),
                    0,
                    self.img.size[0],
                    black_area,
                    y_projection=y_projection
                )
                lines.append(line)
                start_pos = None
                y_projection = []
                black_area = 0
        # deal with edge case when the image does not end with white space
        if y_projection:
            line = Line(
                start_pos,
                start_pos + len(y_projection),
                0,
                self.img.size[0],
                black_area,
                y_projection=y_projection
            )
            lines.append(line)

        return lines

    def splitLines(self):
        lines = []
        for i, line in enumerate(self.lines):
            if line.height < 1.5 * self.stat['baseHeight']:
                lines.append(line)
            else:
                # if a line is very tall, detect if the projection has a minimum
                # around the center. If so, it indicates two lines are wrongly merged into one
                # so we break this line into two lines from the center
                searchRange = 10  # in pixels. Range to search for
                mid = len(line.y_projection)//2
                median = np.median(line.y_projection)
                zero_idx = [i for i, v in enumerate(line.y_projection)
                            if v < 0.15 * median and mid-searchRange < i < mid+searchRange]
                if not zero_idx:
                    # no break point is found
                    lines.append(line)
                # if this line has only two components on the x_projection, it indicates that
                # this line is composed of a fraction line + an operator
                elif self.countXProjComponents(line) == 2:
                    lines.append(line)
                else:
                    breakpoint = int(np.median(zero_idx))
                    lines.extend(line.breakLines(breakpoint))
        self.lines = lines

    def countXProjComponents(self, line):
        # count how many connected components in the x_projection
        # if there are only two connected components, it indicates
        # that this line is composed of a fraction line + an operator
        self.calculateLineXProj(line)
        proj = line.x_projection
        # components = (rising edges + falling edges)/2
        edges = 0
        for i in range(1, len(proj)):
            if (proj[i] > 0 and proj[i-1] == 0) or \
                    (proj[i] == 0 and proj[i - 1] > 0):
                edges += 1
        return edges / 2

    def shrinkLineWidth(self):
        for line in self.lines:
            line.getTrueWidth()

    def pointToLineDist(self, x, y, line):
        # find minimum distance from (x,y) to a line
        # constrain the search region to x-baseHeight to x+baseHeight
        # this has been used for merging hats
        width = self.stat['baseHeight']
        img = self.img.crop((x-width, line.top, x+width, line.bottom))
        pixels = 255 - np.array(img)
        pixels[np.where(pixels > 0)] = 1
        dist = float('inf')
        for i in range(len(pixels)):
            for j in range(len(pixels[0])):
                if pixels[i][j] == 1:
                    # convert index to coordinates
                    y2, x2 = i + line.top, j + x - width
                    dist = min(dist, (x-x2)**2 + (y-y2)**2)
        return math.sqrt(dist)

    def findCloseNeighbor(self, idx):
        # find the neighbor line that are closer to the idx line
        # this has been used for merging hats
        prev, nxt = None, None  # idx of previous line and next line
        dist_prev, dist_nxt = None, None
        if idx == 0:
            nxt = 1
        elif idx == len(self.lines)-1:
            prev = len(self.lines)-1
        else:
            prev, nxt = idx-1, idx+1
        if prev:
            dist_prev = self.pointToLineDist((self.lines[idx].left + self.lines[idx].right)/2,
                                             self.lines[idx].top, self.lines[prev])
        if nxt:
            dist_nxt = self.pointToLineDist((self.lines[idx].left + self.lines[idx].right)/2,
                                            self.lines[idx].bottom, self.lines[nxt])
        # a valid neighbor must not be too far (two times base line gap)
        if (dist_nxt and dist_nxt > 2 * self.stat['baseLineGap'])\
                or dist_nxt == float('inf'):
            dist_nxt = None
        if (dist_prev and dist_prev > 2 * self.stat['baseLineGap'])\
                or dist_prev == float('inf'):
            dist_prev = None
        candidate = None  # if None, no close neighbor exists
        if dist_prev:
            candidate = prev
        if dist_nxt and not dist_prev:
            candidate = nxt
        elif dist_nxt and dist_nxt < dist_prev:
            candidate = nxt
        return candidate

    def findBigOp(self, idx):
        # find the neighbor op line that are closer to the idx line
        # a line may be big op if height > base_height*1.5,
        # line_gap < 2 * baseGap
        prev, nxt = None, None
        gap_prev, gap_nxt = None, None
        if idx != 0:
            prev = idx-1
            gap_prev = self.lines[idx].top - self.lines[prev].bottom
            if self.lines[prev].height < 2.3 * self.lines[idx].height or \
                    gap_prev > 2 * self.stat['baseLineGap']:
                prev = None
        if idx != len(self.lines) - 1:
            nxt = idx+1
            gap_nxt = self.lines[nxt].top - self.lines[idx].bottom
            if self.lines[nxt].height < 2.3 * self.lines[idx].height or \
                    gap_nxt > 2 * self.stat['baseLineGap']:
                nxt = None
        if not nxt:
            return prev
        elif not prev:
            return nxt
        else:
            return prev if gap_prev < gap_nxt else nxt

    def mergeSmallLines(self):
        # merge hats and big operator variables
        lines = []
        skipNext = False
        for i, line in enumerate(self.lines):
            if skipNext:
                skipNext = False
                continue
            # detect small variable hats.
            # a line is a tiny symbol if area < 2% of median area
            if line.black_area < 0.02 * self.stat['baseArea']:
                neighbor = self.findCloseNeighbor(i)
                if not neighbor:
                    lines.append(line)
                    continue
                if neighbor == i-1:  # merge to previous line
                    lines[-1].mergeLine(line)
                    self.calculateLineXProj(lines[-1])
                else:  # merge to next line
                    self.lines[neighbor].mergeLine(line)
                    self.calculateLineXProj(self.lines[neighbor])
            # independent black line is candidate for fraction line
            elif line.black_area/(line.height*line.width) == 1:
                # if this line is longer than neighbor lines and
                # close to neighbor lines, then it is a fraction line
                # merge this line with its neighbors
                if i != 0 and i != len(self.lines)-1 and\
                        line.left <= self.lines[i-1].left and\
                        line.left <= self.lines[i+1].left and\
                        line.right >= self.lines[i-1].right and\
                        line.right >= self.lines[i+1].right and\
                        self.lines[i+1].top - line.bottom < self.stat['baseLineGap'] and\
                        line.top - self.lines[i - 1].bottom < self.stat['baseLineGap']:
                    # old method: merge fraction line
                    lines[-1].mergeLine(line)
                    lines[-1].mergeLine(self.lines[i+1])
                    # # new method: do not merge fraction line. Translate then separately
                    # # and then merge together. This gives better results
                    # lines[-1].fraction = True
                    # self.calculateLineXProj(lines[-1])
                    # lines.append(self.lines[i+1])
                    # lines[-1].fraction = True

                    self.calculateLineXProj(lines[-1])
                    skipNext = True
                else:
                    lines.append(line)
            # detect binding variables. A candidate should have
            # area < 10% baseArea and height < 70% baseHeight
            elif line.black_area < 0.15 * self.stat['baseArea'] and\
                    line.height < 0.7 * self.stat['baseHeight']:
                opLine = self.findBigOp(i)
                if not opLine:
                    lines.append(line)
                elif opLine == i-1:
                    lines[-1].mergeLine(line)
                    self.calculateLineXProj(lines[-1])
                elif opLine == i+1:
                    self.lines[opLine].mergeLine(line)
                    self.calculateLineXProj(self.lines[opLine])
            else:
                lines.append(line)
        self.lines = lines

    def constructWords(self):
        for i, line in enumerate(self.lines):
            # split the line into words
            line.LineToWords()
            # shrink the white space
            for word in line.words:
                # calculate y projection of a word
                img = self.img.crop((word.left, word.top, word.right, word.bottom))
                word.y_projection = self.calculateProjection(img, 'y')
                word.getTrueHeight()

    def runOCR(self, pdf_folder, filename):
        # in this function we call binary CNN to detect math
        # at the same time we call plaintext ocr engine to get text
        # generate the images
        pdf_path = pdf_folder + filename + '.pdf'
        image_dir = pdf_folder + filename + '/'
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        os.mkdir(tmp_folder)
        source_file = open(tmp_folder + 'src.txt', "w")
        source_file_ocr = open(tmp_folder + 'src_ocr.txt', "w")
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)
        for i, line in enumerate(self.lines):
            for j, word in enumerate(line.words):
                # generate the images for binary CNN classifier
                img_ori = self.img.crop((word.left, word.top, word.right, word.bottom))
                img = downsample_image(img_ori, downsample_ratio)
                PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = padding_size
                new_size = (img.size[0] + PAD_LEFT + PAD_RIGHT, img.size[1] + PAD_TOP + PAD_BOTTOM)
                new_im = Image.new("RGB", new_size, (255, 255, 255))
                new_im.paste(img, (PAD_LEFT, PAD_TOP))
                image_path = image_dir + str(i) + '_' + str(j) + '.png'
                new_im.save(image_path)
                source_file.write(image_path + "\n")

                # generate the images for plaintext OCR. The difference is that
                # these images are not downsampled
                img = img_ori
                PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = padding_size
                new_size = (img.size[0] + PAD_LEFT + PAD_RIGHT, img.size[1] + PAD_TOP + PAD_BOTTOM)
                new_im = Image.new("RGB", new_size, (255, 255, 255))
                new_im.paste(img, (PAD_LEFT, PAD_TOP))
                image_path = image_dir + str(i) + '_' + str(j) + '_ocr.png'
                new_im.save(image_path)
                source_file_ocr.write(image_path + "\n")

                # # Tesseract OCR, which gives text labels to each token
                # img = img_ori
                # PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = padding_size
                # new_size = (img.size[0] + PAD_LEFT + PAD_RIGHT, img.size[1] + PAD_TOP + PAD_BOTTOM)
                # new_im = Image.new("RGB", new_size, (255, 255, 255))
                # new_im.paste(img, (PAD_LEFT, PAD_TOP))
                # text = pytesseract.image_to_string(new_im)
                # word.text = text
        source_file.close()
        source_file_ocr.close()

        # Binary CNN Classifier
        translate.run_binaryCNN_external(pdf_path)
        # Plaintext OCR Engine
        translate.run_plaintextOCR_external(pdf_path)

        with open(pdf_folder + filename + '_pred.txt', 'r') as f:
            me_labels = f.readlines()
        with open(pdf_folder + filename + '_pred_ocr.txt', 'r') as f:
            plaintext = f.readlines()

        # prepare features for CRF
        idx = 0
        for i, line in enumerate(self.lines):
            for j, word in enumerate(line.words):
                word.me_likelihood = float(me_labels[idx][:-1]) #1-float(me_labels[idx][:-1])
                word.me_label = False if word.me_likelihood < 0.5 else True
                # add text
                text = plaintext[idx]
                # get rid of the spaces
                word.text = ''.join([c for i, c in enumerate(text) if i % 2 == 0])
                word.is_plaintext = True if word.text in english_vocab else False
                # word.is_plaintext = True if (word.text in english_vocab and
                #                              word.text not in letter_vocab) else False
                if word.text and word.text.lower() in stopwords:
                    # word.me_label = False
                    word.is_stopword = True
                idx += 1
        os.remove(pdf_folder + filename + "_src.txt")
        os.remove(pdf_folder + filename + "_pred.txt")
        os.remove(pdf_folder + filename + "_pred_ocr.txt")

        # remove images
        try:
            shutil.rmtree(image_dir)
        except:
            print('image deletion failure!')

    def translateLaTeX(self):
        tmp_folder = 'pdfs/temp_data/'
        source_file = open(tmp_folder + 'src.txt', "w")
        if not os.path.exists(tmp_folder):
            os.mkdir(tmp_folder)
        for i, line in enumerate(self.lines):
            for j, word in enumerate(line.words):
                if not word.me_label:
                    continue
                img_ori = self.img.crop((word.left, word.top, word.right, word.bottom))
                img = downsample_image(img_ori, downsample_ratio)
                PAD_TOP, PAD_LEFT, PAD_BOTTOM, PAD_RIGHT = padding_size
                new_size = (img.size[0] + PAD_LEFT + PAD_RIGHT, img.size[1] + PAD_TOP + PAD_BOTTOM)
                new_im = Image.new("RGB", new_size, (255, 255, 255))
                new_im.paste(img, (PAD_LEFT, PAD_TOP))
                image_path = tmp_folder + str(i) + '_' + str(j) + '.png'
                new_im.save(image_path)
                source_file.write(image_path + "\n")
        source_file.close()

        # LaTeX Translator
        translate.run_LaTeX_external()
        # read the translation results
        with open(tmp_folder + 'src.txt', 'r') as f:
            files = f.readlines()
            indexes = []
            for file in files:
                line_idx, word_idx = file.split('.')[0].split('/')[-1].split('_')
                indexes.append((int(line_idx), int(word_idx)))
        with open(tmp_folder + 'pred.txt', 'r') as f:
            latex = f.readlines()
            latex = [remove_space_in_latex(l)[:-1] for l in latex]
        # replace the text with the LaTeX contents
        for i in range(len(latex)):
            li, wi = indexes[i]
            self.lines[li].words[wi].text = '$' + latex[i] + '$'

        # remove image folder
        try:
            shutil.rmtree(tmp_folder)
        except:
            print('image deletion failure!')

    def mergeMath(self):
        # merge consecutive math tokens
        for line in self.lines:
            words = []
            math = None
            for word in line.words:
                if word.me_label:
                    if math == None:
                        math = word
                    else:
                        math.mergeMath(word)
                else:
                    if math != None:
                        words.append(math)
                        math = None
                    words.append(word)

            if math != None:
                words.append(math)

            line.words = words

    def postProcessing(self):
        self.mergeMath()
        for i, line in enumerate(self.lines):
            if len(line.words) == 1 and line.words[0].me_label:
                line.IME = True
            else:
                line.IME = False

    def drawYProjection(self):
        # plot the y-projection (lines separator)
        plt.plot(self.y_projection)
        zeros = np.flatnonzero(self.y_projection == 0)
        plt.plot(zeros, self.y_projection[zeros], 'ko')

    def drawLines(self):
        img = self.img.copy()
        draw = ImageDraw.Draw(img)
        # img.convert('RGBA')
        for i, line in enumerate(self.lines):
            draw.rectangle(((line.left, line.top), (line.right, line.bottom)))
            # if line.black_area < 0.02 * self.stat['baseArea']:
            # if line.black_area < 0.1 * self.stat['baseArea'] and \
            #     line.height < 0.7 * self.stat['baseHeight']:
            #     draw.rectangle(((line.left, line.top), (line.right, line.bottom)), fill='black')
        img.show()

    def drawWords(self, text=True):
        img = self.img.copy()
        draw = ImageDraw.Draw(img)
        for i, line in enumerate(self.lines):
            for word in line.words:
                if text:
                    draw.text((word.left-10, word.top-10), str(word.me_likelihood)[:3])
                if word.me_label:
                    draw.rectangle(((word.left, word.top), (word.right, word.bottom)), width=5)
                else:
                    draw.rectangle(((word.left, word.top), (word.right, word.bottom)))
        img.show()

    def drawLabel(self, label):
        # ground truth should be a list of lists
        # each token should have exactly one label
        img = self.img.copy()
        draw = ImageDraw.Draw(img)
        assert (len(label) == len(self.lines))
        for i, line in enumerate(self.lines):
            print(i)
            assert (len(label[i]) == len(self.lines[i].words))
            for j, word in enumerate(line.words):
                if label[i][j] == '1':
                    draw.rectangle(((word.left, word.top), (word.right, word.bottom)), width=5)
                else:
                    draw.rectangle(((word.left, word.top), (word.right, word.bottom)))
        img.show()

    def outputText(self):
        output = ''
        # skipNext = False
        for i, line in enumerate(self.lines):
            # if skipNext:
            #     skipNext = False
            #     continue
            # if line.fraction:
            #     output += '\\begin{equation*}\\frac{' +\
            #               line.words[0].text[1:-1] + \
            #               '}{' +\
            #               self.lines[i+1].words[0].text[1:-1] + \
            #               '}\\end{equation*}' + ' '
            #     skipNext = True
            #     continue
            if not line.IME:
                for word in line.words:
                    output += word.text + " "
            else:
                    output += '\\begin{equation*}' + line.words[0].text[1:-1] + '\\end{equation*}' + ' '
        return output
        # f = open(filename, "w")
        # for line in self.lines:
        #     for word in line.words:
        #         f.write(word.text + " ")
        # f.close()

    @staticmethod
    def splitMathFromText(text):
        # given a string of text, separate the math from text
        # return a list of sentences
        separator = []
        sentences = []
        for i in range(len(text)):
            if text[i] == '$':
                separator.append(i)
        assert(len(separator) % 2 == 0)
        idx = 0
        for i in range(int(len(separator)/2)):
            sentences.append(text[idx:separator[i*2]])
            sentences.append(text[separator[i*2]:separator[i*2+1]+1])
            idx = separator[i*2+1]+1
        sentences.append(text[idx:-1])

        return sentences

    @staticmethod
    def spellcheck(text):
        text = re.sub(r'- ', '', text)
        text = re.sub('mathcal{', 'cal{', text)
        text = re.sub(r'\\begin{equation\*?}', '$', text)
        text = re.sub(r'\\end{equation\*?}', '$', text)
        text = re.sub(r'\\begin{align\*?}', '$', text)
        text = re.sub(r'\\end{align\*?}', '$', text)
        text = re.sub(r'\\begin{center}', '$', text)
        text = re.sub(r'\\end{center}', '$', text)
        text = re.sub(r'\\begin{eqnarray\*?}', '$', text)
        text = re.sub(r'\\end{eqnarray\*?}', '$', text)
        text = re.sub(r'\\begin{gather\*?}', '$', text)
        text = re.sub(r'\\end{gather\*?}', '$', text)
        text = re.sub(r'\\begin{alignat\*?}', '$', text)
        text = re.sub(r'\\end{alignat\*?}', '$', text)
        text = re.sub(r'\\begin{multline}', '$', text)
        text = re.sub(r'\\end{multline}', '$', text)
        sentences = ImageProfile.splitMathFromText(text)
        text = ''
        for i, s in enumerate(sentences):
            if s == '':
                continue
            if s[0] == '$':
                text += s
            else:
                words = s.split(' ')
                for word in words:
                    if word == '':
                        continue
                    # handle non-char part in the word
                    s, e = 0, len(word)-1
                    while s < len(word) and not word[s].isalpha():
                        s += 1
                    while e+1 > s and not word[e].isalpha():
                        e -= 1
                    if s >= e:
                        text += word + ' '
                        continue
                    pre = word[:s]
                    word_new = word[s:e+1]
                    post = word[e+1:]
                    if not word_new.isupper():
                        corrected = spell.correction(word_new)
                    else:
                        corrected = word_new
                    text += pre + corrected + post + ' '
        return text

    @staticmethod
    def calculateProjection(pil_img, axis_option):
        # axis_option: 'x' or 'y'
        # binarize the image and calculate the projection
        # black => 1, white => 0
        pixels = 255 - np.array(pil_img)
        pixels[np.where(pixels > 0)] = 1
        if axis_option == 'y':
            projection = pixels.sum(axis=1)
        else:  # 'x'
            projection = pixels.sum(axis=0)
        return projection

    @staticmethod
    def PDF2PIL(pdf_path, resolution):
        with(wandimage(filename=pdf_path, resolution=resolution)) as source:
            source.alpha_channel = False
            img_buffer = np.asarray(bytearray(source.make_blob(format='png')), dtype='uint8')
        bytesio = io.BytesIO(img_buffer)
        pil_img = Image.open(bytesio)
        pil_img = pil_img.convert('L')
        return pil_img


class Line:
    def __init__(self, top, bottom, left, right,
                 black_area, x_projection=None, y_projection=None):
        self.top = top  # top position of the line, inclusive
        self.bottom = bottom  # bottom position of the line, exclusive
        self.left = left  # inclusive
        self.right = right  # exclusive
        self.black_area = black_area  # area of the black pixels of this line
        self.height = bottom - top
        self.width = right - left
        self.x_projection = x_projection  # used for word segmentation
        self.y_projection = y_projection  # used for line segmentation
        # self.fraction = False  # true indicates that this line is part of a fraction

    def getTrueWidth(self):
        non_zeros = np.where(self.x_projection != 0)
        self.left = non_zeros[0][0]
        self.right = non_zeros[0][-1] + 1
        self.width = self.right - self.left

    def breakLines(self, breakpoint):
        # break a line that are wrongly merged together
        line1 = Line(self.top, self.top + breakpoint,
                     self.left, self.right, sum(self.y_projection[:breakpoint]),
                     y_projection=self.y_projection[:breakpoint])
        line2 = Line(self.top + breakpoint, self.bottom,
                     self.left, self.right, sum(self.y_projection[breakpoint:]),
                     y_projection=self.y_projection[breakpoint:])
        return [line1, line2]

    def mergeLine(self, line):
        # merge a line into this line
        self.top = min(self.top, line.top)
        self.bottom = max(self.bottom, line.bottom)
        self.left = min(self.left, line.left)  # inclusive
        self.right = max(self.right, line.right)  # exclusive
        self.black_area = self.black_area + line.black_area
        # reset projections
        self.x_projection = None
        self.y_projection = None

    def LineToWords(self):
        self.words = []
        # components is a list of connected components on x_projection
        # each element is represented as a tuple (left, right) -> (inclusive, exclusive)
        components = []
        component = None
        for idx in range(self.left, self.right):
            if self.x_projection[idx] != 0:
                if not component:
                    component = [idx, idx]
                else:
                    component[1] += 1
            else:
                if not component:
                    continue
                component[1] += 1  # make the right boundary exclusive
                components.append(tuple(component))
                component = None
        if component:
            component[1] += 1
            components.append(tuple(component))
        # calculate gaps between components
        gaps = [None] * (len(components)-1)
        # edge case: only one word
        if not gaps:
            self.words.append(Word(self.top, self.bottom, components[0][0], components[0][1]))
            return
        for i in range(len(components)-1):
            gaps[i] = components[i+1][0] - components[i][1] + 1
        # assume the most common gap is the gap of characters within a word
        # character_gap = Counter(gaps).most_common(1)[0][0]
        character_gap = np.quantile(gaps, 0.25)
        # assume word gap is twice the character gap
        word_gap = 2 * character_gap
        for i in range(len(components)):
            word = Word(self.top, self.bottom, components[i][0], components[i][1])
            if not self.words or word.left - self.words[-1].right > word_gap:
                self.words.append(word)
            else:
                self.words[-1].mergeWord(word)


class Word:
    def __init__(self, top, bottom, left, right):
        self.top = top  # top position of the word, inclusive
        self.bottom = bottom  # bottom position of the word, exclusive
        self.left = left  # inclusive
        self.right = right  # exclusive
        self.height = bottom - top
        self.width = right - left
        self.y_projection = None
        self.me_label = None
        self.me_likelihood = None
        self.text = None
        self.is_stopword = False

    def mergeWord(self, word):
        self.left = min(self.left, word.left)
        self.right = max(self.right, word.right)

    def getTrueHeight(self):
        non_zeros = np.where(self.y_projection != 0)
        # if non_zeros[0][0] != 0:
        top = self.top
        self.top = top + non_zeros[0][0]
        self.bottom = top + non_zeros[0][-1] + 1
        self.height = self.bottom - self.top

    def mergeMath(self, word):
        self.top = min(self.top, word.top)
        self.bottom = max(self.bottom, word.bottom)
        self.left = min(self.left, word.left)
        self.right = max(self.right, word.right)
        self.height = self.bottom - self.top
        self.width = self.right - self.left
        self.y_projection = None
        self.me_likelihood = None
        self.text = None


def columnDetectionLPF(pil_img):
    # detect if the page has two columns
    # apply low pass filter to the x-projection
    # return a list of splitted images
    width, height = pil_img.size
    # calculate the x-axis projection to detect double column
    x_projection = ImageProfile.calculateProjection(pil_img, 'x')
    # low-pass FIR filter to smooth projection profile for noise handling
    taps = signal.firwin(9, 0.125)
    x_smooth = signal.filtfilt(taps, 1.0, x_projection)
    # thresholding to zero
    threshold = 0.1
    x_smooth[np.where(x_smooth < max(x_smooth) * threshold)] = 0
    lbound, rbound = 0.3, 0.7  # range in x-axis percentage to search for middle break
    zero_idx = [i for i, v in enumerate(x_smooth)
                if v == 0 and x_smooth.size * lbound < i < x_smooth.size * rbound]
    middle_cut = None
    if len(zero_idx) > 10:
        middle_cut = int(np.mean(zero_idx))
    if middle_cut:
        img_left = pil_img.crop((0, 0, middle_cut, height))
        img_right = pil_img.crop((middle_cut, 0, width, height))
        images = [img_left, img_right]
    else:
        images = [pil_img]
    return images


# plt.plot(x_projection,color='0.7',linewidth=1.5,label="Raw")
# plt.plot(x_smooth,'k',linewidth=2,label="LPF")
# plt.legend(loc="upper right", prop={'size': 42})
# plt.show()