# input is a pdf file
# output is a folder containing all the individual pages in pdf format
# output page number start at 1 instead of 0
# return the number of pages
from PyPDF2 import PdfFileReader, PdfFileWriter
import os

pdf_name = 'mon_s_sample'

def split_pdf_into_pages(pdf_path):


	if not os.path.exists(pdf_name):
		os.makedirs(pdf_name)

	with open(pdf_path, 'rb') as infile:

		reader = PdfFileReader(infile)
		for pid in range(reader.getNumPages()):
			writer = PdfFileWriter()
			writer.addPage(reader.getPage(pid))
			with open(pdf_name + '/' + pdf_name + '_'
					  + str(pid) + '.pdf', 'wb') as outfile:
				writer.write(outfile)

split_pdf_into_pages('E:\zelun/' + pdf_name + '.pdf')