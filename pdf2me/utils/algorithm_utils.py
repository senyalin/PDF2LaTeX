from pdfminer.layout import LTChar
from pdf_parser.pdfbox_wrapper import get_pdf_page_size
import numpy as np
from collections import Counter
import re
import math


def get_chunk_width_ratio(chunk):
    # if end with (xx), return ratio 1
    if re.match('.+\(\d+\)', chunk['text']):
        return 1
    # calculate the ratio of char_len / char_len+space_len
    # first convert the width into 0.1-wide bins
    width = round(chunk['bbox'].right(), 1) - round(chunk['bbox'].left(), 1)
    bins = set()
    for symbol in chunk['symbols']:
        left = round(symbol.bbox[0], 1)
        right = round(symbol.bbox[2], 1)
        while left <= right:
            bins.add(left)
            left += 0.1
    width_ratio = len(bins) / (width*10)

    return width_ratio


def sort_and_merge_overlap(char_list_list):
    """
    merge accent, will try to merge each accent symbol with word first
    :param word_info_list:
    :param fontname2space:
    :param char_list_list:
    :return:
        still return list of char_list,
        but merge the vertical overlapping ones
    """
    if len(char_list_list) == 0:
        return char_list_list

    from utils.box_utils import linesToBoxes, merge_overlap_lines, split_by_space_char_list_list
    line_boxes = linesToBoxes(char_list_list)
    # sort the lines according to vertical positions
    char_list_list = [x for x, _ in sorted(zip(char_list_list, line_boxes),
                                                key=lambda pair: pair[1].top(), reverse=True)]

    # merge lines that are overlapped vertically
    lines_num = len(char_list_list)
    while True:
        # merge overlapped vertical lines multiple times until converge
        char_list_list = merge_overlap_lines(char_list_list)
        if lines_num == len(char_list_list):
            break
        lines_num = len(char_list_list)
    # the last step of splitting by the space, as the space is kept
    char_list_list = split_by_space_char_list_list(char_list_list)

    return char_list_list


def remove_empty_lines(char_list_list):
    """
    empty means only have LTAnno in them
    :param char_list_list:
    :return:
    """
    new_char_list_list = []
    for char_list in char_list_list:
        if is_empty_line(char_list):
            continue
        new_char_list_list.append(char_list)
    return new_char_list_list


def is_empty_line(char_list):
    for c in char_list:
        if isinstance(c, LTChar) and c.get_text() not in [' ', 'space']:
            return False
    return True


def split_double_column(pdf_path, char_list_list):
    # deal with double column here
    # sort lines horizontally and vertically, then merge overlapped lines
    if is_double_column(pdf_path, char_list_list):
        from utils.box_utils import get_char_list_bbox
        double_column_margin = 15
        # split the current list into three parts if detected as double column
        # outside of the double column, left column, and right column
        page_size = get_pdf_page_size(pdf_path)
        page_width = page_size['width']

        out_char_list_list = []
        left_char_list_list = []
        right_char_list_list = []
        for char_list in char_list_list:
            bbox = get_char_list_bbox(char_list)
            if bbox.left() < bbox.right() < page_width / 2 + double_column_margin:
                left_char_list_list.append(char_list)
            elif bbox.right() > bbox.left() > page_width / 2 - double_column_margin:
                right_char_list_list.append(char_list)
            else:
                out_char_list_list.append(char_list)

        new_out_char_list_list = sort_and_merge_overlap(out_char_list_list)
        new_left_char_list_list = sort_and_merge_overlap(left_char_list_list)
        new_right_char_list_list = sort_and_merge_overlap(right_char_list_list)

        left_bound = [get_column_left(new_left_char_list_list),
                      get_column_left(new_right_char_list_list)]

        # not in the vertical range of the double column
        # center on the left part,
        # center on the right part,
        char_list_list = []
        char_list_list.extend(new_out_char_list_list)
        char_list_list.extend(new_left_char_list_list)
        char_list_list.extend(new_right_char_list_list)
        res_char_list_list = char_list_list
    else:
        # single column, then just go on merging the lines
        res_char_list_list = sort_and_merge_overlap(char_list_list)
        left_bound = [get_column_left(res_char_list_list)]

    return res_char_list_list, left_bound


def get_column_left(column):
    lbounds = [None] * len(column)
    for i, line in enumerate(column):
        lbounds[i] = math.floor(line[0].bbox[0])
    return Counter(lbounds).most_common(1)[0][0]  # most frequent bound


def is_double_column(pdf_path, char_list_list):
    """
        The idea is that if there are two cluster of begin position , then double column

    :param pdf_path:
    :param pid:
    :return:
    """
    page_size = get_pdf_page_size(pdf_path)
    page_width = page_size['width']

    # get the boundary of the column, collect the startpoint, and end point, use 0.95 quantile
    start_pos_list = []
    end_pos_list = []
    quantile = 0.90
    from utils.box_utils import get_char_list_bbox
    for char_list in char_list_list:
        # remove line with less than 10 chars
        if len(char_list) < 30:
            continue
        bbox = get_char_list_bbox(char_list)
        start_pos_list.append(bbox.left())
        end_pos_list.append(bbox.right())

    if len(start_pos_list) == 0 or len(end_pos_list) == 0:
        # it's an empty page.
        return False

    start_pos = np.percentile(start_pos_list, int((1-quantile)*100))
    end_pos = np.percentile(end_pos_list, int(quantile*100))

    if end_pos < page_width / 2 or start_pos > page_width/2:
        # if only half of the column have enough lines.
        return True

    center_pos = (start_pos+end_pos)/2
    good_line_count = 0
    total_count = 0.0

    for char_list in char_list_list:
        # remove line with less than 10 chars
        if len(char_list) < 30:
            continue

        bbox = get_char_list_bbox(char_list)
        if bbox.left() < bbox.right() < center_pos or \
                bbox.right() > bbox.left() > center_pos:
            good_line_count += 1
        total_count += 1

    threshold = 0.6
    if float(good_line_count) / total_count > threshold:
        return True
    else:
        return False