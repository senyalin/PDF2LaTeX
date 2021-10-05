from pdfminer.layout import LTChar
from plotting.draw_bbox_to_pdf import draw_filtered_bbox_to_pdf
from utils.consts import special_nme_glyph_list
from utils.separate_math_text import check_is_math_LTChar, check_by_font_name

def findMESymbolsFromList(char_list_list, pdf_path):
    lt_char_list = []
    for char_list in char_list_list:
        for char in char_list:
            if isinstance(char, LTChar):
                lt_char_list.append(char)
    candidates_filter = findMESymbolCandidates(lt_char_list)
    draw_filtered_bbox_to_pdf(lt_char_list, pdf_path, candidates_filter)


def findMESymbolCandidates(lt_char_list):
    # attach me_candidate and me_likelihood to LTChars
    # likelihood ratio test: if a char is a math symbol
    # count the occurrence of each font size, by considering the length of unchanged substring
    threshold = 3  # if font size remains the same for longer than this threshold, we count for long. Otherwise, we count for short
    count_long = {}
    count_short = {}

    prev = ''
    counter = 0
    for i in range(len(lt_char_list)):
        curr = lt_char_list[i].size
        if curr not in count_short:
            count_short[curr] = 0
        if curr not in count_long:
            count_long[curr] = 0
        if curr == prev:
            counter += 1
        else:
            if prev != '':
                if counter <= threshold:
                    count_short[prev] += counter
                else:
                    count_long[prev] += counter
            prev = curr
            counter = 1
    if counter != 0:
        if counter <= threshold:
            count_short[prev] += counter
        else:
            count_long[prev] += counter

    print 'Bayesian stats'
    print count_short
    print count_long

    me_candidates_index = [0]*len(lt_char_list)
    for i in range(len(lt_char_list)):

        if count_long[lt_char_list[i].size] < count_short[lt_char_list[i].size]:
            me_candidates_index[i] += 1 # 1 to denote char detected from size
        if check_is_math_LTChar(lt_char_list[i]):
            me_candidates_index[i] += 2  # 2 to denote char detected from math lib
        # 3 to denote both
        if any(text in special_nme_glyph_list for text in
               [lt_char_list[i].raw_text] + [lt_char_list[i]._text]):
            me_candidates_index[i] = 0

        if me_candidates_index[i] > 0:
            lt_char_list[i].me_candidate = True
        else:
            lt_char_list[i].me_candidate = False
        # likelihood is between 0 and 1
        lt_char_list[i].me_likelihood = 1 \
            if count_long[lt_char_list[i].size] == 0 or lt_char_list[i].me_candidate else \
            1.0 * count_short[lt_char_list[i].size] / (count_short[lt_char_list[i].size] + count_long[lt_char_list[i].size])
        # print lt_char_list[i].me_likelihood
        # if lt_char_list[i].me_likelihood != 0 and lt_char_list[i].me_likelihood != 255:
        #     print lt_char_list[i].me_likelihood
    return me_candidates_index
