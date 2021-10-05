import os
import re
import pickle
import string
from utils.name2latex import normal_fence_mapped
import copy
from pdfminer.layout import LTChar
from utils.math_resources import unicode2latex, unicode2gn, str2gn
from utils.name2latex import normal_fence_mapped, name2latex


def pdfbox_parser(pdf_path):
    EXPORT_CHAR_JAR_NAME = "pdf2me/pdf_parser/PrintFontBoxGlyphBox.jar"

    cmd = "java -jar {} {}".format(EXPORT_CHAR_JAR_NAME, pdf_path)
    os.system(cmd)


def pdf_extract_chars_helper(pdf_path):
    """
    return list of chars as well as their glyph
    :return:
        list of LTChars based on the layout analysis of the
        each LTChar is also associated with the glyph names,

        The priority for the name: glyph -> unicode value -> code
    """
    page_size = get_pdf_page_size(pdf_path)

    lt_char = pickle.load(open('pdf2me/utils/one_ltchar.pkl', "rb"))

    lt_char_list = []
    char_list = []
    fontsize_list = []
    lines = get_exported_char_lines(pdf_path)

    for line in lines:
        if line.startswith("CHARINFO"):
            ws = line.strip().split('\t')
            # if there is a non digit one, keep the one
            # otherwise could only inference based on the digit
            if len(ws) < 6:
                continue
            val = get_char_val(ws, line)
            # might be the value is \t
            bbox = None
            fontbox = None
            for tmpi in range(5, len(ws)):
                if ws[tmpi].count(',') == 3:
                    bbox = [float(v) for v in ws[tmpi].split(',')]
                    break

            for tmpi in range(7, len(ws)):
                if ws[tmpi].count(',') == 3:
                    fontbox = [float(v) for v in ws[tmpi].split(',')]
                    break

            if len(bbox) != 4 or len(fontbox) != 4:
                raise Exception('bbox error')

            # adjust the box position
            new_top = page_size['height'] - bbox[1]
            new_bottom = page_size['height'] - bbox[3]
            bbox[1], bbox[3] = new_bottom, new_top
            new_top = page_size['height'] - fontbox[1]
            new_bottom = page_size['height'] - fontbox[3]
            fontbox[1], fontbox[3] = new_bottom, new_top

            # use the original parenthesis, for mathcing of citation
            if val in normal_fence_mapped.keys():
                # ws[3]  # the unicode value part
                val = normal_fence_mapped[val]

            c = create_char(
                lt_char,
                simplify_glyph(val),
                ws[1],
                bbox)
            c.fontbox = fontbox

            dy = fontbox[3] - fontbox[1]
            dx = fontbox[2] - fontbox[0]
            dy_g = bbox[3] - bbox[1]
            dx_g = bbox[2] - bbox[0]
            v_h_thres = 0.8  # threshold to determine if the char is vertical
            if dx == 0:
                c.size = format(dy, '.3f')
            else:
                c.size = format(dy if dy/dx > v_h_thres else dx, '.3f')
            if dx_g == 0:
                c.glyph_size = format(dy_g, '.3f')
            else:
                c.glyph_size = format(dy_g if dy_g/dx_g > v_h_thres else dx_g, '.3f')
            c.raw_text = ws[3]
            if ws[3] == 'null':
                if c.get_text().isdigit():
                    c.raw_text = c.get_text()
            # need some transformation to make it right
            if c.bbox[0] == c.bbox[2] or c.bbox[1] == c.bbox[3] or\
                    c.get_text() == ' ':
                # throw away bad symbols using bbox_size == 0
                continue
            lt_char_list.append(c)
            char_list.append(ws[3])
            fontsize_list.append(c.size)

    return lt_char_list, char_list, fontsize_list


def get_pdf_page_size(pdf_path):
    page_char_path = pdf_path + '.0.txt'
    with open(page_char_path) as f:
        for line in f:
            if line.startswith("PDFINFO"):
                m = re.search(r"\(([\d\.]+), ([\d\.]+)\)", line)
                res_dict = {'width': float(m.group(1)), 'height': float(m.group(2))}
                return res_dict
    raise Exception("fail to get the PDFINFO")


def get_exported_char_lines(pdf_path):
    page_char_path = pdf_path + '.0.txt'
    lines = []
    with open(page_char_path) as f:
        for line in f:
            lines.append(line)
    return lines


def get_char_val(ws, line):
    """
    :param ws2: is the code
    :param ws3: is the possible ASCII
    :param ws4: is the glyph name
    :return:
    """
    val = None
    if val is None and ws[4] not in ["", 'null']:
        val = ws[4]
    elif (val is None) and (ws[3] not in ["", 'null']):
        val = ws[3]
    elif val is None and ws[2] not in ["", 'null']:
        if int(ws[2]) in range(256):
            val = chr(int(ws[2]))
        else:
            val = ""
        # only after the glyph name and unicode not valid
        # try to do such mapping.
        if "TimesNewRoman" in ws[1] or "Arial" in ws[1]:
            # https: // en.wikipedia.org / wiki / Windows - 1252
            i2u = {
                146: 'quoteright',
                150: 'dash',
            }
            int_val = int(ws[2])
            if int_val in i2u:
                val = i2u[int_val]
            else:
                assert int_val < 128
                val = chr(int_val)
    try:
        if isinstance(val, unicode) and val in unicode2latex:
            val = unicode2latex[val]
        elif isinstance(val, str) or isinstance(val, unicode):
            if isinstance(val, unicode):
                val = val.encode('utf-8', 'ignore')
            valid_char_list = string.letters
            valid_char_list += string.whitespace
            valid_char_list += string.digits
            valid_char_list += string.printable
            if val not in valid_char_list:
                if val in str2gn:
                    val = str2gn[val]
                else:
                    tmp_uni_val = val.decode('utf-8')
                    if tmp_uni_val in unicode2latex:
                        val = unicode2latex[tmp_uni_val]
                    elif tmp_uni_val in unicode2gn:
                        val = unicode2gn[tmp_uni_val]
                    elif tmp_uni_val in name2latex:
                        val = tmp_uni_val
                    elif tmp_uni_val in ['ffi', 'fi', 'ff', 'fl']:
                        val = tmp_uni_val
                    elif re.match(r'[a-z]\d+', tmp_uni_val):
                        val = "graphics"
                    else:
                        # print "single uval unicode {}".format(tmp_uni_val.encode('raw_unicode_escape'))
                        # print tmp_uni_val
                        # pdf_util_error_log.error(line)
                        # val = "badwindows"
                        pass
    except:
        val = "badwindows"  # just set as empty
    return val


def create_char(char, val, fontname=None, bbox=None):
    res_char = copy.copy(char)
    res_char._text = val
    if fontname is not None:
        res_char.fontname = fontname
    if bbox is not None:
        res_char.set_bbox(bbox)
    return res_char


def simplify_glyph(val):
    """
    the glyph is only for non english chars
    """
    # TODO, the bracket, parenthesis, brace as curve bracket

    name2char = {
        'period': '.',
        'comma': ',',
        'colon': ':',
        'space': ' ',

        'zero': '0',
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',

        #'equal': '=',
        #'greater': '>',
        #'less': '<',
    }
    if val in name2char:
        return name2char[val]
    if val in normal_fence_mapped:
        return normal_fence_mapped[val]
    return val