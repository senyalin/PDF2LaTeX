from utils.consts import left_parentheses, right_parentheses
import math
import re
from utils.algorithm_utils import get_chunk_width_ratio


# split leading and ending comma, period, and questionmark
split_symbols_rule = [',', '.', ';', '?']
IME_regex_right = '.*\(\d+(\.\d+)*\)$'
IME_regex_left = '^\(\d+(\.\d+)*\).*'
theorem_regex = '(theorem|lemma|proof|definition) *(\d(\.\d+)?)?(\.|:)?'
section_regex = '^\d+(\.\d+)?(\.\d+)? *[A-Za-z]+$'


# merge spatially close chunks with a threshold
def merge_spatial_chunks_rule(chunk_list, thres):
    from utils.groupChunk import merge_chunks
    new_chunk_list = []
    merge_pool = []
    # merge spatially close chunks
    for i in range(len(chunk_list) - 1):
        merge_pool.append(chunk_list[i])
        if chunk_list[i + 1]['bbox'].left() - chunk_list[i]['bbox'].right() <= thres:
            continue
        new_chunk_list.append(merge_chunks(merge_pool))
        merge_pool = []
    merge_pool.append(chunk_list[-1])
    new_chunk_list.append(merge_chunks(merge_pool))
    return new_chunk_list


# merge unmathced parenthesis within one line
def valid_parenthesis_rule(chunk_list):
    from utils.groupChunk import merge_chunks, get_chunk_struct
    merge_pool = []
    new_chunk_list = []
    i = 0
    ime = len(chunk_list) == 1
    while i < len(chunk_list):
        chunk = chunk_list[i]
        # remove the leading parentheses of an ME
        if chunk['me_candidate'] and len(chunk['symbols']) > 1 and not ime\
                and chunk['text'][0] == '(' and not chunk['text'][1].isdigit():
            new_chunk_list.append(get_chunk_struct([chunk['symbols'][0]]))
            chunk = get_chunk_struct(chunk['symbols'][1:])
        merge_pool.append(chunk)
        if chunk['me_candidate']:
            pcount = [0] * len(left_parentheses)  # parentheses_count
            pcount = count_unmatched_parentheses(chunk, pcount)
            while not valid_parentheses_match_rule(pcount):
                i += 1
                if i >= len(chunk_list):
                    if not valid_parentheses_match_rule(pcount):
                        return chunk_list
                    break
                chunk = chunk_list[i]
                merge_pool.append(chunk)
                pcount = count_unmatched_parentheses(chunk, pcount)
        new_chunk_list.append(merge_chunks(merge_pool))
        merge_pool = []
        i += 1

    return new_chunk_list


def count_unmatched_parentheses(chunk, parentheses_count):
    for i, p in enumerate(left_parentheses):
        parentheses_count[i] += chunk['text'].count(p)
    for i, p in enumerate(right_parentheses):
        parentheses_count[i] -= chunk['text'].count(p)
    return parentheses_count


def valid_parentheses_match_rule(parentheses_count):
    # count matches such as '(]'
    return sum(parentheses_count) <= 0
    # return all(i <= 0 for i in parentheses_count)


def EME_on_left_bound_rule(merged_chunk, hist_info, left_bound):
    # deal with double column left bound
    if len(left_bound) == 1:
        lbound = left_bound[0]
    else:
        df1 = abs(math.floor(merged_chunk['bbox'].left()) - left_bound[0])
        df2 = abs(math.floor(merged_chunk['bbox'].left()) - left_bound[1])
        lbound = left_bound[0] if df1 < df2 else left_bound[1]
    if abs(math.floor(merged_chunk['bbox'].left()) - lbound) <= 1\
            and (merged_chunk['bbox'].right() -
                 merged_chunk['bbox'].left())/hist_info['common_width'] < 0.8 and \
            not re.match(IME_regex_left, merged_chunk['text']):
        return True
    return False


def is_line_ime_rule(line, hist_info, left_bound):
    if len(line) == 1:
        chunk = line[0]
        if chunk['me_candidate']:
            if EME_on_left_bound_rule(chunk, hist_info, left_bound):
                return False
            # if indexed, it must be an IME
            from groupChunk import end_with_parenthesis
            if end_with_parenthesis(chunk, hist_info):
                return True
            # 1) too few characters indicates that it's probably a wrong detection
            # 2) probably a paragraph title if alpha ratio is too large
            if get_chunk_width_ratio(chunk) < 0.2 or (chunk['alpha_num'] > 5 and chunk['alpha_ratio'] > 0.8)\
                    or 'http' in chunk['text'] or 'php' in chunk['text']:
                chunk['me_candidate'] = False
            else:
                return True
    return False
