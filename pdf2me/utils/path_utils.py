import os
from pdfminer.layout import LTChar, LTAnno
import xml.etree.ElementTree as ET
import string
from utils.bbox import BBox
import struct
import sys
import re
from utils.name2latex import name2latex
from utils.math_resources import get_latex_commands, special_unicode_chars, unicode2latex


def get_filename_from_folder(pdf_name, pdf_folder):
    return pdf_folder + '/' + pdf_name + '.pdf'


def is_space_char(char):
    if isinstance(char, LTAnno):
        return True
    if isinstance(char, LTChar) and char.get_text() in [' ', 'space']:
        return True
    return False


def get_file_name_prefix(file_path):
    """

    :param file_path:
    :return:
    """
    file_name = get_file_name(file_path)
    if '.' in file_name:
        return file_name[:file_name.rindex('.')]
    return file_name


def get_file_name(file_path):
    """

    :param file_path:
    :return:
    """
    file_path = file_path.replace("\\", "/")
    file_name = file_path[file_path.rindex("/")+1:]
    return file_name


def double2hexlongbits(double):
    i = struct.unpack('Q', struct.pack('d', double))
    # print i, format(i, 'x')
    res = format(int(i[0]), 'x')
    return res


def icst_bbox2str(bbox):
    if isinstance(bbox, BBox):
        res = "%s %s %s %s" % (
            double2hexlongbits(bbox.left()),
            double2hexlongbits(bbox.top()),
            double2hexlongbits(bbox.right()),
            double2hexlongbits(bbox.bottom())
        )
    else:
        # the str is in the order of left, top, right, bottom
        res = "%s %s %s %s"%(
            double2hexlongbits(bbox[0]),
            double2hexlongbits(bbox[3]),
            double2hexlongbits(bbox[2]),
            double2hexlongbits(bbox[1])
            )
    return res


def readable_bbox2str(bbox):
    """
    bbox as left, bottom, right, top

    :param bbox:
    :return:
    """
    if isinstance(bbox, BBox):
        res = " ".join([str(v) for v in bbox.to_list()])
    else:
        res = " ".join([str(v) for v in bbox])
    return res


def export_xml(page_info, out_path):
    page_n = ET.Element('Page', {'PageNum': str(page_info['pid'])})
    from utils.box_utils import get_char_list_bbox
    # mergesort ime and eme by id
    i = j = 0
    while i < len(page_info['iid']) or j < len(page_info['eid']):
        if j == len(page_info['eid']) or (i != len(page_info['iid']) and
                int(page_info['iid'][i]) < int(page_info['eid'][j])):
        # for i, ime_line in enumerate(page_info['ilist']):
            ime_line = page_info['ilist'][i]
            bbox = get_char_list_bbox(ime_line)

            i_n = ET.SubElement(
                page_n,
                'IsolatedFormula',
                {
                    'BBox': readable_bbox2str(bbox),
                    'MathIndex': page_info['idxlist'][i],
                    'id': page_info['iid'][i]
                })
            for box in page_info['boxlist'][i]:
                c_n = ET.SubElement(
                    i_n,
                    'LineBox',
                    {
                        'Box': readable_bbox2str(box),
                    })
            for char in ime_line:
                if isinstance(char, LTChar):
                    clean_text = get_latex_val_of_lt_char(char)
                    clean_text = invalid_xml_remove(clean_text)
                    c_n = ET.SubElement(
                        i_n,
                        'Char',
                        {
                            'BBox': readable_bbox2str(char.bbox),
                            'FSize': str(char.size),
                            'GSize': str(char.glyph_size),
                            'Text': clean_text
                        })
            i += 1
        elif i == len(page_info['iid']) or (j != len(page_info['eid']) and
                int(page_info['iid'][i]) > int(page_info['eid'][j])):
            # the eme part
            # for i, eme in enumerate(page_info['elist']):
            eme = page_info['elist'][j]
            bbox = get_char_list_bbox(eme)

            i_n = ET.SubElement(
                page_n,
                'EmbeddedFormula',
                {
                    'BBox': readable_bbox2str(bbox),
                    'id': page_info['eid'][j]
                })
            for char in eme:
                if isinstance(char, LTChar):
                    clean_text = get_latex_val_of_lt_char(char)
                    clean_text = invalid_xml_remove(clean_text)
                    c_n = ET.SubElement(
                        i_n,
                        'Char',
                        {
                            'BBox': readable_bbox2str(char.bbox),
                            'FSize': str(char.size),
                            'GSize': str(char.glyph_size),
                            'Text': clean_text
                        })
            j += 1
    try:
        res = ET.tostring(page_n, encoding='utf-8')
        if out_path:
            with open(out_path, 'w') as f:
                print>>f, res
        else:
            print res

    except Exception as e:
        print e


def invalid_xml_remove(c):
    #http://stackoverflow.com/questions/1707890/fast-way-to-filter-illegal-xml-unicode-chars-in-python
    illegal_unichrs = [ (0x00, 0x08), (0x0B, 0x1F), (0x7F, 0x84), (0x86, 0x9F),
                    (0xD800, 0xDFFF), (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF),
                    (0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), (0x3FFFE, 0x3FFFF),
                    (0x4FFFE, 0x4FFFF), (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                    (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), (0x9FFFE, 0x9FFFF),
                    (0xAFFFE, 0xAFFFF), (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                    (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), (0xFFFFE, 0xFFFFF),
                    (0x10FFFE, 0x10FFFF) ]

    illegal_ranges = ["%s-%s" % (unichr(low), unichr(high))
                  for (low, high) in illegal_unichrs
                  if low < sys.maxunicode]

    illegal_xml_re = re.compile(u'[%s]' % u''.join(illegal_ranges))
    bad_char_list = ['\xb0', '\xb6', '\xb2', '\xab', '\xac', '\xaa', '\xba']

    # remove larger than >= 128
    # not quite sure here, assume all ascii not UTF-8
    clean_larger_than_128 = ""
    for one_c in c:
        one_c_val = ord(one_c)
        if one_c_val >= 128:
            continue
        clean_larger_than_128 += one_c
    c = clean_larger_than_128

    for bad_char in bad_char_list:
        try:
            c = c.replace(bad_char, '')  # lead to encoding error
        except Exception as e:

            print 'ignore back value', c
            print e

    try:
        c.decode('utf-8')
    except Exception as e:
        print "TODO log the error", e
        return ' '

    if illegal_xml_re.search(c) is not None:
        #Replace with space
        return ' '
    else:
        return c


def get_latex_val_of_lt_char(c):
    """
    The latex value is determined based on the combination of information
    from the latex command and the glyph name
    """
    assert isinstance(c, LTChar)
    gn = c.get_text()
    return get_latex_val_of_gn(gn)


def get_latex_val_of_gn(gn):
    if gn.startswith("\\"):
        return gn

    if gn is None:
        pdf_util_error_log.error("Failed to get the glyph name for {}".format(c))
        raise Exception("Could not get glyph Name")

    if gn in string.lowercase:
        return gn
    if gn in string.uppercase:
        return gn
    if gn in ['-', '%', '.', ';', ',', ':']: # some normal chars
        return gn
    if len(gn) == 1 and ord(gn) < 128:
        return gn

    if gn == "ignore" or gn == "badwindows":
        return "bad_val"

    latex_command = get_latex_commands()
    if gn in latex_command:
        return "\\"+gn
    if gn in name2latex:
        return name2latex[gn]

    # NOTE: to be abandoned, or just separate the math from the other
    if gn in special_unicode_chars:
        return gn

    if isinstance(gn, unicode):
        if gn in unicode2latex:
            return unicode2latex[gn]

    elif isinstance(gn, str):
        tmp_gn = gn
        if tmp_gn in unicode2latex:
            return unicode2latex[tmp_gn]

    # TODO, some special chars?
    if gn in ['circlecopyrt']:
        if gn == 'circlecopyrt':
            return 'R'
        return gn

    if gn.endswith('script'):
        letter = gn[:-6]
        if len(letter) == 1:
            return "\\mathcal{{{}}}".format(letter)

    if re.match(r"\d+", gn) and int(gn) < 256:
        v = int(gn)
        return chr(v)

    # the font does not provide further information
    if re.match(r"[#A-Za-z]{0,2}\d*", gn):
        return gn

    # bad names should not have latex values
    invalid_gn_for_latex =[
        'null',     # might be pdfbox processing error
        'heart',    # might be the poker symbol
        'section',  # to indicate the reference, not as ME
        '.notdef',  # bad
        'ESC', 'FF', 'VT', 'BS', 'ETB', 'HT', 'SO',

        # arabic character
        'chard5', 'charce', 'charab', 'char0b', 'char58',

        # old english letters
        'Eth',

        # some others
        #https://www.w3.org/TR/MathML2/isonum.html
        'yen', 'trademark', 'currency',
        'guillemotleft', 'guillemotright',

        # TODO,
        'owner', 'squash',

        # special font type
        'weierstrass',

        # TODO, need to check where, they are mostly math concept here
        'power', 'norm', 'abs', 'divide', 'softmax', 'plus-or-minus',
        'notforcesextra', 'nottriangeqlleft', #10.1.1.1.2077_5
        'squareimage', #10.1.1.1.2105_5
        'arrowvertex', 'arrowbt', 'wreathproduct', #10.1.1.138.4863_11
        'twosuperior',  # 10.1.1.193.1818_2
        'ring',  # 10.1.1.6.2202_3
        'satisfies', 'notturnstile', 'forces', 'notforces',  # 10.1.1.6.2246_15
        'squiggleright',  # 10.1.1.6.2302_13
        'ordmasculine',  # 10.1.1.6.2330_13

    ]
    if gn in invalid_gn_for_latex:
        return ''
    if gn in func_name_list:
        return gn

    print gn.encode('hex')
    #raise Exception("Unknown"+gn)
    pdf_util_error_log.error("Unknown "+gn.encode('utf-8', 'ignore'))
    raise UnknownGlyphName(gn)

