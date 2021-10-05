"""
TODO, seems to be a deprecated file, but many function still under use

seperate math from text

The unicode from get_text will not work when the notation don't have a unicode, rather simply glyph name.
First try to solve by update pdfminer, still not working. http://www.unixuser.org/~euske/python/pdfminer/
`make cmap before install pdfminer`

Second, use the pdfbox package.
"""

from nltk.stem.wordnet import WordNetLemmatizer
from pdfminer.layout import LTChar
from utils.consts import math_font_key_list, alphabeta_font_key_list

# TODO, organize the glyph name based approach
gn_me_prob_path = "pdf2me/utils/gn_me_prob.txt"
wnl = WordNetLemmatizer()


def load_me_glyph(fpath=gn_me_prob_path, prob_thres = 0.5):
    """
    There is an artifact, where the glyph name is digit, not right.
    Stat from the ground truth

    :param fpath:
    :param prob_thres:
    :return:
    """
    lines = open(fpath).readlines()
    glyph_list = []
    for line in lines:
        if line.strip() == "":
            continue
        if line.startswith("#"):
            continue

        ws = line.split(" ")
        if ws[0].isdigit():
            continue
        if len(ws[0]) == 1: # a single char
            continue
        if float(ws[1]) > prob_thres:
            glyph_list.append(ws[0])
    return glyph_list


# NOTE, TODO, this list to be extended at needed
glynameList = load_me_glyph()
expanded_glyph_list = [
    "summationdisplay",
    "Delta",
    "plus",
    "angbracketleft",
    "angbracketright",
    "defines",

    "element",

    # in paper 2246
    "notturnstile",
    "turnstileleft",
    "forces",
    "precedesorcurly",
    "Gamma",

    # some symbol level
    "+",
    ">",
    "<"
]
glynameList.extend(expanded_glyph_list)


def check_is_math_LTChar(c, pdfbox_font=None):
    """
    first is based on unicode
    the second is based on the glyph names

    :param c:
    :param pdfbox_font:
    :return:
    """
    # speical rules
    if pdfbox_font:
        glyph_name = get_char_glyph(c, pdfbox_font)
        if glyph_name in ['bullet', 'hyphen']:
            return False

    c_str = c.get_text()
    if isinstance(c_str, unicode) and check_is_math(c_str):
        return True

    if pdfbox_font:
        if check_glyph(c, pdfbox_font):
            return True
    else:
        # if the font is empty, should be from pdfbox
        gn = c.get_text()
        if gn in glynameList:
            return True

    if check_by_font_name(c):
        return True
    return False


def check_by_font_name(c):
    if not isinstance(c, LTChar):
        return False

    for pn in math_font_key_list:
        if c.fontname.count(pn) > 0:
            return True

    text = c.get_text()
    if text.isalpha():
        for pn in alphabeta_font_key_list:
            if c.fontname.count(pn) > 0:
                return True
    return False


def check_glyph(c, pdfbox_font):
    """
    check based on the glyph name
    :param c:
    :param pdfbox_font:
    :return:
    """
    # based on the font name, unicode, map back to the glyname
    if pdfbox_font:
        glyph_name = get_char_glyph(c, pdfbox_font)
        if glyph_name in glynameList:
            return True
    return False


def check_is_math(c):
    #http://unicode.org/charts/
    #http://unicode.org/charts/PDF/U2200.pdf
    math_unicode_range = [u'\u2200', u'\u22ff'] #\u2212
    #http://unicode.org/charts/PDF/U0370.pdf
    greek_unicode_range = [u'\u0370', u'\u03ff']
    #http://unicode.org/charts/PDF/U2070.pdf
    #http://unicode.org/charts/PDF/U2190.pdf
    arrow_unicode_range = [u'\u2190', u'\u21ff']

    if c >= math_unicode_range[0] and c <= math_unicode_range[1]:
        return True
    if c >= greek_unicode_range[0] and c <= greek_unicode_range[1]:
        return True
    if c >= arrow_unicode_range[0] and c <= arrow_unicode_range[1]:
        return True
    if c == '=':
        return True
    if c.count('cid') > 0:
        return True

    return False


