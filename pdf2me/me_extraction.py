'''
 * Copyright (c) 2017-2019, Texas Engineering Experiment Station (TEES), a
 * component of the Texas A&M University System.
 * All rights reserved.
 * The information and source code contained herein is the exclusive
 * property of TEES and may not be disclosed, examined or reproduced
 * in whole or in part without explicit written authorization from TEES.
'''


from pdf_parser.pdf_split import split_pdf_into_pages
from utils.me_extraction_utils import extract_me
import sys


def main():
    # first argument: pdf path
    path = sys.argv[1]
    # second argument: a pdf file. can be multiple pages
    pdf_name = sys.argv[2]
    # parse pdf into individual pages
    page_num = split_pdf_into_pages(path + pdf_name + '.pdf')
    pdf_folder = path + pdf_name
    # process multiple page documents
    for pn in range(page_num):
        pdf_path = pdf_folder + '/' + pdf_name + '_p' + str(pn) + '.pdf'
        me = extract_me(pdf_path)


if __name__ == "__main__":
    main()


# process single page dataset
# pdf_folder = 'test_files'
# pdfs = os.listdir(pdf_folder)
# failed_files = []
#
# import time
# time1 = time.time()
# for i, pdf in enumerate(pdfs):
#     if not pdf.endswith(".pdf"):
#         continue
#     try:
#         print pdf
#         extract_me(pdf_folder+'/'+pdf)
#     except:
#         failed_files.append(pdf)
# print failed_files
# time2 = time.time()
# print 'time cost: '
# print time2 - time1