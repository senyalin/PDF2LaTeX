
# input is a pdf file
# output is a folder containing all the individual pages in pdf format
# output page number start at 1 instead of 0
# return the number of pages
from PyPDF2 import PdfFileReader, PdfFileWriter
from utils.path_utils import get_file_name_prefix
import os

def split_pdf_into_pages(pdf_path):
	reader = PdfFileReader(file(pdf_path, 'rb'))

	if os.path.isdir(os.path.splitext(pdf_path)[0]):
		print 'pdf is already splitted'
		return reader.getNumPages()

	pfd_name = get_file_name_prefix(pdf_path)
	pdf_folder = pdf_path[:-4]

	if not os.path.exists(pdf_folder):
		os.makedirs(pdf_folder)
	for pid in range(reader.getNumPages()):
		writer = PdfFileWriter()
		writer.addPage(reader.getPage(pid))
		writer.write(file(pdf_folder+'/'+pfd_name+'_p'+str(pid)+'.pdf', 'wb'))

	return reader.getNumPages()
# split_pdf_into_pages('E:\zelun\plotting/lda2vec.pdf')