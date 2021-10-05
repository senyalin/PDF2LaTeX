from pdfminer.layout import LTChar, LTAnno
import numpy as np
from utils.bbox import BBox
from utils.path_utils import is_space_char
from utils.math_resources import accent_name_list
from collections import Counter
from utils.math_resources import func_name_list
from plotting.draw_bbox_to_pdf import draw_bbox_list_to_pdf
from path_utils import get_latex_val_of_lt_char


def get_char_list_bbox(char_list):
    tmp_char_list = []
    for char in char_list:
        if not isinstance(char, LTChar):
            continue
        tmp_char_list.append(char)

    left_list = [char.bbox[0] for char in tmp_char_list]
    right_list = [char.bbox[2] for char in tmp_char_list]
    bottom_list = [char.bbox[1] for char in tmp_char_list]
    top_list = [char.bbox[3] for char in tmp_char_list]
    if len(left_list) == 0 or\
        len(right_list) == 0 or\
        len(bottom_list) == 0 or\
        len(top_list) == 0:
        print "WARNING: no bbox for a empty char list"
        return BBox([0, 0, 0, 0])
    new_bbox = (
        np.min(left_list),
        np.min(bottom_list),
        np.max(right_list),
        np.max(top_list))
    return BBox(new_bbox)


# get the bbox of lines
def linesToBoxes(char_list_list):
    line_boxes = [None] * len(char_list_list)
    for i in range(len(char_list_list)):
        line_boxes[i] = get_char_list_bbox(char_list_list[i])
    return line_boxes


def merge_overlap_lines(char_list_list):
    # get the bbox of lines
    line_boxes = linesToBoxes(char_list_list)
    new_char_list_list = []
    merge_pool = []  # store lines to be merged
    for i in range(len(char_list_list) - 1):
        merge_pool.append(char_list_list[i])
        line_space = line_boxes[i].bottom() - line_boxes[i + 1].top()
        # print line_space
        if line_space <= 0.001 and line_space > -400:
            # if line_boxes[i].v_overlap(line_boxes[i+1]):
            print 'overlapped on line: ' + str(i)
            # merge_pool.append(char_list_list[i])
        else:
            new_char_list_list.append(merge_lines(merge_pool))
            merge_pool = []
    merge_pool.append(char_list_list[-1])
    new_char_list_list.append(merge_lines(merge_pool))
    return new_char_list_list


# merge multiple lines of LTChars and sort them from left to right
def merge_lines(lines):
    ## comment the following in order to sort every line
    char_list = []
    for line in lines:
        for c in line:
            if isinstance(c, LTChar):
                char_list.append(c)
    char_list.sort(key=lambda char_arg: char_arg.bbox[0])

    return char_list


def split_by_space_char_list_list(char_list_list):
    new_char_list_list = []
    for char_list in char_list_list:
        new_char_list = []
        for char in char_list:
            if is_space_char(char):
                if len(new_char_list) > 0 and not is_space_char(new_char_list[-1]):
                    new_char_list.append(LTAnno(" "))
            else:
                new_char_list.append(char)
        new_char_list_list.append(new_char_list)
    return new_char_list_list


def insert_sapce_and_merge_lines(char_list_list, pdf_path):

    # draw_bbox_list_to_pdf(line_boxes, pdf_path)
    # the accent line only merge with the following line
    char_list_list = merge_accent_line(char_list_list)

    line_boxes = linesToBoxes(char_list_list)
    hist_info = histogram2stat(char_list_list, line_boxes)

    # merge the bind var
    lines_len = len(char_list_list)
    char_list_list = merge_big_operators(char_list_list, line_boxes, hist_info)
    if lines_len != len(char_list_list):
        line_boxes = linesToBoxes(char_list_list)

    # insert space (LTAnno) between words using an adaptive threshold
    new_char_list_list = []
    for char_list in char_list_list:
        new_char_list_list.append(
            insert_space_using_fontbox(char_list, hist_info['word_interval']))

    draw_bbox_list_to_pdf(line_boxes, pdf_path)
    return new_char_list_list, hist_info, line_boxes


def merge_accent_line(char_list_list):
    new_char_list_list = []
    merge_pool = []
    for i, char_list in enumerate(char_list_list):
        merge_pool.append(char_list)
        if only_accent(char_list):
            continue
        else:
            new_char_list_list.append(merge_lines(merge_pool))
            merge_pool = []
    if merge_pool:
        new_char_list_list.append(merge_lines(merge_pool))
    # line_boxes = linesToBoxes(new_char_list_list)
    # draw_bbox_list_to_pdf(line_boxes, pdf_path)
    return new_char_list_list


def only_accent(char_list):
    # check for accent
    is_only_accent = True
    from utils.math_resources import accent_name_list
    for c in char_list:
        if isinstance(c, LTChar) and c.get_text() not in accent_name_list:
            is_only_accent = False
    return is_only_accent


def histogram2stat(char_list_list, line_boxes):
    # new method to calculate histogram and find spacing between words and lines
    char_spaces = []
    line_spaces = [0] # first line default to 0
    # because it's the space between -1 line and 0 line which does not exist
    line_heights = []
    line_widths = []
    line_common_heights = []
    for i in range(len(char_list_list)):
        common_heights = []
        # sort the lines here
        char_list_list[i].sort(key=lambda char_arg: char_arg.bbox[0])
        # calculate line height
        line_height = line_boxes[i].top() - line_boxes[i].bottom()
        line_height = round(line_height)
        line_heights.append(line_height)
        line_width = line_boxes[i].right() - line_boxes[i].left()
        line_width = round(line_width)
        line_widths.append(line_width)
        # print 'line height: ' + str(line_height)
        # calculate line spacing
        if i > 0:
            line_space = line_boxes[i - 1].bottom() - line_boxes[i].top()
            line_space = round(line_space * 2) / 2
            # print 'line space: ' + str(line_space)
            line_spaces.append(line_space)
        # calculate char spacing
        for j in range(0, len(char_list_list[i])):
            if j > 0:
                char_space = char_list_list[i][j].fontbox[0] - char_list_list[i][j - 1].fontbox[2]
                char_space = max(0, round(char_space * 2) / 2)  # round to the nearest 0.5, non-negative
                char_spaces.append(char_space)
            common_heights.append((char_list_list[i][j].fontbox[3]-char_list_list[i][j].fontbox[1]))
        common_height = Counter(line_heights).most_common(1)[0][0]  # most common line height
        line_common_heights.append(common_height)
    # word_intervals = Counter(char_spaces).most_common(2)
    # word_interval = (word_intervals[0][0] + word_intervals[1][0]) / 2 # the medium number
    word_interval = 1

    line_height = Counter(line_heights).most_common(1)[0][0]  # most common line height
    line_space = Counter(line_spaces).most_common(1)[0][0]  # most common line interval
    line_width = Counter(line_widths).most_common(1)[0][0]  # most common line width
    hist_info = {}
    hist_info['word_interval'] = word_interval
    hist_info['heights'] = line_heights
    hist_info['spaces'] = line_spaces
    hist_info['widths'] = line_widths
    hist_info['common_height'] = line_height
    hist_info['common_space'] = line_space
    hist_info['common_width'] = line_width
    hist_info['avg_width'] = sum(hist_info['widths']) / len(hist_info['widths'])
    hist_info['text_heights'] = line_common_heights
    return hist_info


def merge_big_operators(char_list_list, line_boxes, hist_info):
    operator_list = ['\\sum', '\\prod', '\\int', '\\wedge', '\\vee']
    # index of lines with big operators
    big_operator_line_idx = []
    for i in range(len(char_list_list)):
        for char in char_list_list[i]:
            latex_val = get_latex_val_of_lt_char(char)
            if latex_val in operator_list:
                big_operator_line_idx.append(i)
                break
    # neighbors/bind_var candidates of big operator lines
    # a candidate neighbor line has to be bounded by a big operator line
    neighbors = {}
    # the dict key indicates that the key line is a neighbor
    # the dict val includes all candidates that it could bind to
    for big_op_line in big_operator_line_idx:
        prev_line = big_op_line - 1
        next_line = big_op_line + 1
        if prev_line >= 0 and \
                is_bind_var_line(prev_line, big_op_line, line_boxes, hist_info, char_list_list, None):
            if prev_line not in neighbors:
                neighbors[prev_line] = []
            neighbors[prev_line].append(big_op_line)
        if next_line < len(char_list_list) and \
                is_bind_var_line(next_line, big_op_line, line_boxes, hist_info, char_list_list, None):
            if next_line not in neighbors:
                neighbors[next_line] = []
            neighbors[next_line].append(big_op_line)

    # find closest big_op line to bind to
    merge_candidates = []
    for key, big_lines in neighbors.iteritems():
        min_dist = float('inf')
        for big_line in big_lines:
            curr_dist = vdist_between_two_lines(key, big_line, line_boxes)
            if curr_dist < min_dist:
                candidate_line = big_line
                min_dist = curr_dist
        # if a line is in candidates, it indicates that this line need to be merged with its next line
        merge_candidates.append(min(key, candidate_line))

    new_char_list_list = []
    merge_pool = []
    for i, char_list in enumerate(char_list_list):
        merge_pool.append(char_list)
        if i in merge_candidates:
            continue
        new_char_list_list.append(merge_lines(merge_pool))
        merge_pool = []
    if merge_pool:
        new_char_list_list.append(merge_lines(merge_pool))

    return new_char_list_list


def is_bind_var_line(bind_line, op_line, line_boxes, hist_info, char_list_list, chunk_list_list):
    # determine if the line is a candidate for bind_var line
    bind_width = line_boxes[bind_line].right() - line_boxes[bind_line].left()
    op_width = line_boxes[op_line].right() - line_boxes[op_line].left()
    symbol_spatial_ratio = 1
    symbol_width = 0
    if char_list_list:
        for symbol in char_list_list[bind_line]:
            symbol_width += symbol.bbox[2] - symbol.bbox[0]
        symbol_spatial_ratio = symbol_width / bind_width
    if chunk_list_list:
        for chunk in chunk_list_list[bind_line]:
            for symbol in chunk['symbols']:
                symbol_width += symbol.bbox[2] - symbol.bbox[0]
        symbol_spatial_ratio = symbol_width / bind_width
    return \
        line_boxes[bind_line].left() + 0.1*op_width > line_boxes[op_line].left() and \
        line_boxes[bind_line].right() < line_boxes[op_line].right() and \
        (bind_width < op_width/2 or symbol_spatial_ratio < 0.55) and \
        (hist_info['heights'][bind_line] < hist_info['common_height']*0.8 or
         vdist_between_two_lines(bind_line, op_line, line_boxes) < hist_info['common_space']*1.5) and \
         not vdist_between_two_lines(bind_line, op_line, line_boxes) > hist_info['common_height']


def vdist_between_two_lines(line1, line2, line_boxes):
    # assume two lines do not overlap, but the order of two lines is unknown
    return min(
        abs(line_boxes[line1].top() - line_boxes[line2].bottom()),
        abs(line_boxes[line2].top() - line_boxes[line1].bottom()))





def insert_space_using_fontbox(char_list_in, thres):
    """
    segment
    :param char_list_in: the list of char to insert LTAnno
    """
    char_list_in = [c for c in char_list_in if isinstance(c, LTChar)]
    return_char_list = []
    for cid, char in enumerate(char_list_in):
        is_space = False
        if isinstance(char, LTChar) and char.get_text() in [' ', 'space']:
            is_space = True
        if is_space:
            return_char_list.append(char)  # NOTE, still keep the space char for later analysis
            return_char_list.append(LTAnno(' '))
            continue
        return_char_list.append(char)
        # if the space between two fontbox is less than a threshold, insert a space
        if cid != len(char_list_in) - 1:
            interval = char_list_in[cid+1].fontbox[0] - char_list_in[cid].fontbox[2]
            # print interval
            if interval >= thres:
                return_char_list.append(LTAnno(' '))
    # remove extra ltaano
    final_return_char_list = []
    for char in return_char_list:
        if len(final_return_char_list) > 0 and \
                isinstance(char, LTAnno) and \
                isinstance(final_return_char_list[-1], LTAnno):
            continue
        final_return_char_list.append(char)
    return final_return_char_list