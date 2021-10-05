# This function draws bbox on top of PDF files

# Original code from https://stackoverflow.com/questions/1180115/add-text-to-existing-pdf-using-python
# https://www.reportlab.com/docs/reportlab-reference.pdf
from PyPDF2 import PdfFileWriter, PdfFileReader
import StringIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
import struct
import string
import xml.dom.minidom
import pickle
import re
from pdf_parser.pdfbox_wrapper import get_pdf_page_size
import skimage.io as sio
from utils.consts import PLOT_MODE
from PIL import Image


def hexlongbits2double(str):
    return struct.unpack('d', struct.pack('Q', int(str, 16)))[0]

def bbox2rect(bbox):
    rect = {}
    rect['l'] = hexlongbits2double(bbox[0])
    rect['t'] = hexlongbits2double(bbox[1])
    rect['r'] = hexlongbits2double(bbox[2])
    rect['b'] = hexlongbits2double(bbox[3])
    area = (rect['r'] - rect['l']) * (rect['t'] - rect['b'])
    return {'area':area, 'rect':rect}

def parse_xml(p_xml):
	flag = True
	try:
		xmldoc = xml.dom.minidom.parse(p_xml)
	except:
		try:
			# Try to replace unprintable chars and parse via string
			f = file(p_xml, 'rb')
			s = f.read()
			f.close()

			ss = s.translate(None, string.printable)
			s = s.translate(None, ss)

			xmldoc = xml.dom.minidom.parseString(s)
		except:
			xmldoc = None
			flag = False
	return flag, xmldoc

def get_info(p_xml):
    formulas = []
    flag, xmldoc = parse_xml(p_xml)
    if flag:
        # foreach all the Embedded Formulas
        for i in xmldoc.getElementsByTagName('EmbeddedFormula'):
            formula = {}
            formula.update(bbox2rect(i.getAttribute('BBox').split())) # update rect and area info
            formula['type'] = 'E' # add formula type info
            formulas.append(formula)
        # foreach all the Isolated Formulas
        for i in xmldoc.getElementsByTagName('IsolatedFormula'):
            formula = {}
            formula.update(bbox2rect(i.getAttribute('BBox').split())) # update rect and area info
            formula['type'] = 'I' # add formula type info
            formulas.append(formula)
    return flag, formulas


def getOffset(offset_file):
	with open(offset_file) as f:
		for line in f:
			if line.startswith("PageOffsetInfo"):
				m = re.search(r"\(([\d\.]+), ([\d\.]+)\)", line)
				res_dict = {'x': float(m.group(1)), 'y': float(m.group(2))}
				return res_dict


def draw_gt_bbox_to_pdf(file_name):
	if not PLOT_MODE:
		return
	pdf_file = 'E:\zelun\km_test_data\marmot_math_formula_dataset_v1.0\Dataset\pdf/' + file_name + '.pdf'
	gt_file = 'E:\zelun\km_test_data\marmot_math_formula_dataset_v1.0\Dataset\ground truth/' + file_name + '.xml'
	offset_file = 'E:\zelun\knowledge_mining_data/tmp/test_tmp/10/' + file_name + '/' + file_name + '.pdf.0.txt'

	gt_info = get_info(gt_file)
	gt_info = gt_info[1]

	offset = getOffset(offset_file)

	print('drawing bbox to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)

	## plot original ground truth
	for box in gt_info:
		upperLeft_x = box['rect']['l'] + offset['x']
		upperLeft_y = box['rect']['b'] + offset['y']
		height = box['rect']['r'] - box['rect']['l']
		width = box['rect']['t'] - box['rect']['b']
		if box['type'] == 'I': # green for IME
			can.setStrokeColorRGB(0, 255, 0)
		if box['type'] == 'E':  # blue for EME
			can.setStrokeColorRGB(0, 0, 255)
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	# plot missing boxes Xing added
	can.setStrokeColorRGB(255, 0, 0)
	try:
		with open('xinglabel/'+file_name+'.jpg.fig.label.pkl', 'rb') as f:
			missing_box = pickle.load(f)

		pdf_page_size_info = get_pdf_page_size(pdf_file, 0)

		jpg_path = 'E:\zelun\km_test_data\marmot_math_formula_dataset_v1.0\Dataset\image/' + file_name + '.tif'
		im = Image.open(jpg_path, mode="r")
		width, h = im.size
		# scale xing's ground truth according to the image size / pdf size
		scale = pdf_page_size_info['height'] / h


		bbox_list = [[a[0], h - a[3], a[1], h - a[2]] for a in missing_box]

		scale_bbox_list = []
		for bbox in bbox_list:
			scale_bbox_list.append([v * scale for v in bbox])

		for box in scale_bbox_list:
			upperLeft_x = box[0]
			upperLeft_y = box[1]
			height = box[2] - box[0]
			width = box[3] - box[1]
			can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)
	except:
		pass

	can.save()
	#move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	page.mergePage(new_pdf.getPage(0))
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0]+"_gt_boxes.pdf", "wb")
	output.write(outputStream)
	outputStream.close()



#################################### MAIN ###########################
# files = os.listdir('E:\zelun\km_test_data\marmot_math_formula_dataset_v1.0\Dataset\pdf')
#
# for file_name in files:
# 	file_name = file_name[:-4]
# 	if not file_name[-1].isdigit():
# 		continue
#
# 	print file_name
# 	# file_name = '10.1.1.192.1804_16'
#
# 	draw_gt_bbox_to_pdf(file_name)
