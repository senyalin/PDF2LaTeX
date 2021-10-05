Author:
Zelun Wang

Description:
This is a new independent code repo for math expressions extraction.
The repo implements the font-size based ME extraction algorithm.

Required python libraries:
- PyPDF2
- reportlab
- pdfminer
- nltk
- cylp (optional)

Input:
- A PDF document (single or multiple pages)
Should be placed in the "test_files" folder.

Output:
- PDF pages with highlighted ME BBox
- An XML file with all ME information
Should be saved to the "test_files/pdf_name" folder.