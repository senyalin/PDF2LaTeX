# This function draws bbox on top of PDF files

# Original code from https://stackoverflow.com/questions/1180115/add-text-to-existing-pdf-using-python
# https://www.reportlab.com/docs/reportlab-reference.pdf
from PyPDF2 import PdfFileWriter, PdfFileReader
import StringIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os
from utils.consts import PLOT_MODE
from utils.rules import is_line_ime_rule


def export_words_to_txt(chunk_list_list, txt_path):
	with open(txt_path, "w") as text_file:
		me_id = 0
		for line in chunk_list_list:
			for word in line:
				if not word['me_candidate']:
					text_file.write(word['text'] + ' ')
				else:
					word['me_id'] = str(me_id)
					text_file.write(wrap_me_id(me_id) + ' ')
					me_id += 1
			text_file.write('\n')
	return


def wrap_me_id(id):
	return '$me_id_' + str(id) + '_me_id$'


def draw_full_bbox_to_pdf(char_list, pdf_file, glyphBox=True, fontBox=True):
	if not PLOT_MODE:
		return
	print('drawing full bbox to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)
	for charinfo in char_list:
		#margin = 0.5
		if glyphBox:
			# draw glyph bbox
			upperLeft_x = charinfo.bbox[0]
			upperLeft_y = charinfo.bbox[1]
			height = charinfo.bbox[2] - charinfo.bbox[0]
			width = charinfo.bbox[3] - charinfo.bbox[1]
			can.setStrokeColorRGB(0, 255, 0)
			#can.rotate(90)
			can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)
		if fontBox:
			# draw font bbox
			upperLeft_x = charinfo.fontbox[0]
			upperLeft_y = charinfo.fontbox[1]
			height = charinfo.fontbox[2] - charinfo.fontbox[0]
			width = charinfo.fontbox[3] - charinfo.fontbox[1]
			can.setStrokeColorRGB(255, 0, 0)
			can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)
			
	can.save()
	#move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0]+"_full_boxes.pdf", "wb")
	output.write(outputStream)
	outputStream.close()


def draw_filtered_bbox_to_pdf(char_list, pdf_file, candidates_filter):
	if not PLOT_MODE:
		return
	print('drawing _1_step to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)
	for i in range(len(candidates_filter)):
		if candidates_filter[i] == 0:
			continue # do not draw bbox
		if candidates_filter[i] == 1:
			can.setStrokeColorRGB(255, 0, 0) # red for font size detection
		elif candidates_filter[i] == 2:
			can.setStrokeColorRGB(0, 255, 0) # green for math lib detection
		elif candidates_filter[i] == 3:
			can.setStrokeColorRGB(0, 0, 255) # blue for both
		charinfo = char_list[i]
		# draw font bbox
		upperLeft_x = charinfo.fontbox[0]
		upperLeft_y = charinfo.fontbox[1]
		height = charinfo.fontbox[2] - charinfo.fontbox[0]
		width = charinfo.fontbox[3] - charinfo.fontbox[1]
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.save()
	# move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0] + "_1_step_symbols.pdf", "wb")
	output.write(outputStream)
	outputStream.close()


def drawing_symbol_to_pdf(char_list, pdf_file):
	if not PLOT_MODE:
		return
	print('drawing step 1: symbol level...')
	# green indicates non-ME candidates
	# red indicates ME candidates
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)
	for i in range(len(char_list)):
		if char_list[i].me_candidate:
			can.setStrokeColorRGB(255, 0, 0)  # red for ME
		else:
			can.setStrokeColorRGB(0, 255, 0)  # green for non-ME
		charinfo = char_list[i]
		# draw font bbox
		upperLeft_x = charinfo.bbox[0]
		upperLeft_y = charinfo.bbox[1]
		height = charinfo.bbox[2] - charinfo.bbox[0]
		width = charinfo.bbox[3] - charinfo.bbox[1]
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.save()
	# move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0] + "_1_step.pdf", "wb")
	output.write(outputStream)
	outputStream.close()


def draw_bbox_list_to_pdf(word_box_list, pdf_file):
	if not PLOT_MODE:
		return
	print('drawing lines to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)
	for word_box in word_box_list:
		# draw word bbox
		# left_list = [char.bbox[0] for char in tmp_char_list]
		# right_list = [char.bbox[2] for char in tmp_char_list]
		# bottom_list = [char.bbox[1] for char in tmp_char_list]
		# top_list = [char.bbox[3] for char in tmp_char_list]

		upperLeft_x = word_box.left()
		upperLeft_y = word_box.bottom()
		height = word_box.right() - word_box.left()
		width = word_box.top() - word_box.bottom()
		can.setStrokeColorRGB(0, 255, 0)
		# can.rotate(90)
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.save()
	# move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0] + "_lines.pdf", "wb")
	output.write(outputStream)
	outputStream.close()


def draw_chunk_bbox_to_pdf(chunk_list_list, pdf_file, left_bound, hist_info, postfix=None, finalOutput=False):
	if not PLOT_MODE:
		return
	if postfix:
		print('drawing ' + postfix + ' to pdf...')
	else:
		print('drawing chunk bbox to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)

	for chunk_list in chunk_list_list:
		ime = False
		if is_line_ime_rule(chunk_list, hist_info, left_bound):
			ime = True
		for chunk in chunk_list:
			upperLeft_x = chunk['bbox'].left()
			upperLeft_y = chunk['bbox'].bottom()
			height = chunk['bbox'].right() - chunk['bbox'].left()
			width = chunk['bbox'].top() - chunk['bbox'].bottom()
			if ime:
				can.setStrokeColorRGB(0, 0, 255)
			elif chunk['me_candidate']:
				can.setStrokeColorRGB(255, 0, 0)
			else:
				can.setStrokeColorRGB(0, 255, 0)

			if finalOutput:
				if ime:
					can.setStrokeColorRGB(0, 255, 0)
				elif chunk['me_candidate']:
					can.setStrokeColorRGB(0, 0, 255)
				else:
					continue
			# can.rotate(90)
			can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.save()
	# move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	if not output:
		outputStream = file(os.path.splitext(pdf_file)[0] + "_chunk_boxes.pdf", "wb")
	else:
		outputStream = file(os.path.splitext(pdf_file)[0] + postfix + ".pdf", "wb")
	output.write(outputStream)
	outputStream.close()


def draw_compare_bbox_to_pdf(pdf_file, gt_formulas, md_formulas):
	# draw the ground truth and prediction together
	# ground truth as green
	# prediction as yellow
	if not PLOT_MODE:
		return
	print('drawing compare bbox to pdf...')
	packet = StringIO.StringIO()
	# create a new PDF with Reportlab
	can = canvas.Canvas(packet, pagesize=letter)

	can.setStrokeColorRGB(0, 255, 0)
	for formula in gt_formulas:
		upperLeft_x = formula['rect']['l']
		upperLeft_y = formula['rect']['b']
		height = formula['rect']['r'] - formula['rect']['l']
		width = formula['rect']['t'] - formula['rect']['b']
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.setStrokeColorRGB(255, 255, 0)
	for formula in md_formulas:
		upperLeft_x = formula['rect']['l']
		upperLeft_y = formula['rect']['b']
		height = formula['rect']['r'] - formula['rect']['l']
		width = formula['rect']['t'] - formula['rect']['b']
		can.rect(upperLeft_x, upperLeft_y, height, width, fill=0, stroke=1)

	can.save()
	# move to the beginning of the StringIO buffer
	packet.seek(0)
	new_pdf = PdfFileReader(packet)
	# read your existing PDF
	existing_pdf = PdfFileReader(file(pdf_file, "rb"))
	output = PdfFileWriter()
	# add the "watermark" (which is the new pdf) on the existing page
	page = existing_pdf.getPage(0)
	try:
		page.mergePage(new_pdf.getPage(0))
	except:
		pass
	output.addPage(page)
	# finally, write "output" to a real file
	outputStream = file(os.path.splitext(pdf_file)[0] + "_compare_boxes.pdf", "wb")
	output.write(outputStream)
	outputStream.close()