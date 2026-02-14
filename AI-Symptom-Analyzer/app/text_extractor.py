import PyPDF2
from docx import Document
import pytesseract
from PIL import Image

def from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text=""
    for page in reader.pages:
        text += page.extract_text()
    return text

def from_docx(file):
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def from_image(file):
    img = Image.open(file)
    return pytesseract.image_to_string(img)
