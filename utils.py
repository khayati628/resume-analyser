import re

from docx import Document
from pypdf import PdfReader


def clean_resume(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_text_from_pdf(file):
    reader = PdfReader(file)
    pages = (page.extract_text() or "" for page in reader.pages)
    return "\n".join(pages)


def extract_text_from_docx(file):
    document = Document(file)
    return "\n".join(paragraph.text for paragraph in document.paragraphs)


def extract_text(file, filename):
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "pdf":
        return extract_text_from_pdf(file)
    if extension == "docx":
        return extract_text_from_docx(file)

    raise ValueError(f"Unsupported file type: .{extension}")
