from pdf_parser.pdfbox_wrapper import pdfbox_parser, pdf_extract_chars_helper
from utils.bbox import BBox
from utils.box_utils import get_char_list_bbox, insert_sapce_and_merge_lines
from utils.algorithm_utils import remove_empty_lines, split_double_column
from utils.symbol_detection import findMESymbolsFromList
from utils.groupChunk import chars_to_chunks_using_me_labels, merge_consecutive_me_chunks, \
    get_ime_line_idx, merge_ime_bind_var, merge_close_chunks, \
    split_commas, merge_splitted_imes, merge_middle_symbols, separate_equa_num
from plotting.draw_bbox_to_pdf import draw_chunk_bbox_to_pdf, draw_full_bbox_to_pdf, export_words_to_txt
from utils.rules import is_line_ime_rule
from utils.path_utils import get_file_name_prefix, export_xml
import os


def extract_me(pdf_path):
    pdfbox_parser(pdf_path)

    char_list_list = pdf_extract_lines_raw(pdf_path)

    res_char_list_list, left_bound = split_double_column(pdf_path, char_list_list)

    res_char_list_list, hist_info, line_boxes = insert_sapce_and_merge_lines(res_char_list_list, pdf_path)

    # find ME candidates
    findMESymbolsFromList(res_char_list_list, pdf_path)
    # eval_symbol_performance(lines_to_line(res_char_list_list), pdf_path)
    # print_symbol_performance()

    # detect IME here
    chunk_list_list = chars_to_chunks_using_me_labels(res_char_list_list, hist_info)
    draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_2_step')
    chunk_list_list = merge_consecutive_me_chunks(chunk_list_list)
    draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_3_step')
    # chunk_list_list = chunk_label_with_MRF(chunk_list_list)
    # chunk_list_list = merge_consecutive_me_chunks(chunk_list_list)
    # draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_4_step_MRF')
    # # this step find IME: a line is an IME if it contains me_candidate boxes but does not contain non-me words
    chunk_list_list = get_ime_line_idx(chunk_list_list, hist_info, left_bound)
    chunk_list_list = merge_ime_bind_var(chunk_list_list, left_bound, hist_info, line_boxes)
    draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_5_step_IME')
    chunk_list_list = merge_close_chunks(chunk_list_list, hist_info['word_interval'])
    chunk_list_list = split_commas(chunk_list_list, hist_info, left_bound)
    chunk_list_list = merge_middle_symbols(chunk_list_list, hist_info)
    draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_6_step')
    chunk_list_list = merge_splitted_imes(chunk_list_list, hist_info, left_bound)
    draw_chunk_bbox_to_pdf(chunk_list_list, pdf_path, left_bound, hist_info, '_7_step', True)

    #############
    #  export results into eme.xml and ime.xml
    pdf_name = get_file_name_prefix(pdf_path)
    OUTPUT_XML_PATH = os.path.dirname(pdf_path)
    # export to txt
    txt_path = OUTPUT_XML_PATH + '/' + pdf_name + '.txt'
    export_words_to_txt(chunk_list_list, txt_path)
    # both ime and eme: bme
    bme_xml_path = OUTPUT_XML_PATH + '/' + pdf_name + '.xml'
    eme_list = []
    ime_list = []
    multi_box_list = []
    ime_idx_list = []
    eme_id_list = []
    ime_id_list = []
    for chunk_list in chunk_list_list:
        if is_line_ime_rule(chunk_list, hist_info, left_bound):
            # chunk_list[0], equa_num = separate_equa_num(chunk_list[0])
            ime_chunk, idx = separate_equa_num(chunk_list[0])
            ime_list.append(ime_chunk)
            ime_idx_list.append(idx)
            multi_box_list.append([] if 'ime_bbox' not in chunk_list[0] else chunk_list[0]['ime_bbox'])
            ime_id_list.append(chunk_list[0]['me_id'])
        else:
            for chunk in chunk_list:
                if chunk['me_candidate']:
                    eme_list.append(chunk['symbols'])
                    eme_id_list.append(chunk['me_id'])
    pid = pdf_name[-1]
    bme_list = []
    bme_list.extend(eme_list)
    bme_list.extend(ime_list)
    page_bme = {}
    page_bme['pid'] = pid
    page_bme['boxlist'] = multi_box_list
    page_bme['idxlist'] = ime_idx_list
    page_bme['ilist'] = ime_list
    page_bme['elist'] = eme_list
    page_bme['iid'] = ime_id_list
    page_bme['eid'] = eme_id_list
    export_xml(page_bme, bme_xml_path)

    return res_char_list_list


def pdf_extract_lines_raw(pdf_path):
    """
    each line is a list of LTChar
    based on the order of the original elements.
    """
    lt_char_list, char_list, fontsize_list = pdf_extract_chars_helper(pdf_path)

    draw_full_bbox_to_pdf(lt_char_list, pdf_path, glyphBox=True, fontBox=True)
    # ## find ME candidates
    ## write the data into text files
    # from plotting.findMESymbolCandidates import findMESymbolCandidates
    # me_candidates_index = findMESymbolCandidates(lt_char_list)
    # f = open("E:/zelun/plotting/fontsize.txt", "w+")
    # for i in range(len(char_list)):
    #     f.write(char_list[i] + '  ' + str(fontsize_list[i]) + '  ' + str(me_candidates_index[i]) + '  ' + \
    #             str(lt_char_list[i].fontbox[0]) + '  ' + str(lt_char_list[i].fontbox[1]) + '  ' + \
    #             str(lt_char_list[i].fontbox[2]) + '  ' + str(lt_char_list[i].fontbox[3]) + '\n')
    # f.close()
    #######################################

    # merge chars into word_chunks and lines
    char_list_list = list()
    if len(lt_char_list) == 0:
        return char_list_list
    tmp_char_list = list()

    ## convert text streams into lines
    cur_bbox = BBox(lt_char_list[0].bbox)
    for i, char in enumerate(lt_char_list):
        tmp_bbox = BBox(char.bbox)
        if cur_bbox.v_overlap(tmp_bbox):
            tmp_char_list.append(char)
            cur_bbox = get_char_list_bbox(tmp_char_list)
        else:
            tmp_char_list.sort(key=lambda char_arg: char_arg.bbox[0])
            char_list_list.append(tmp_char_list)
            tmp_char_list = [char]
            cur_bbox = BBox(char.bbox)

    if len(tmp_char_list) > 0:
        char_list_list.append(tmp_char_list)

    # clean the lines with only LTAnno
    char_list_list = remove_empty_lines(char_list_list)
    return char_list_list


