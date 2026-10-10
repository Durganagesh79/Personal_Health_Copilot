import os
import re
from pathlib import Path

import fitz
import pytesseract
from PIL import Image, ImageOps, ImageEnhance


def configure_tesseract():
    """
    If Tesseract is installed but not in PATH on Windows,
    set TESSERACT_CMD in your .env file.
    """
    tesseract_path = os.getenv("TESSERACT_CMD")
    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def extract_from_image(file_path):
    configure_tesseract()

    with Image.open(file_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
        image = ImageEnhance.Contrast(image).enhance(1.5)

        text = pytesseract.image_to_string(
            image,
            config="--psm 6",
        )

    return text.strip()


def extract_from_pdf(file_path):
    pages_text = []

    with fitz.open(file_path) as document:
        for page in document:
            text = page.get_text("text").strip()

            # Use OCR for scanned PDF pages that have little or no text.
            if len(text) < 30:
                configure_tesseract()
                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(2, 2),
                    alpha=False,
                )
                image = Image.frombytes(
                    "RGB",
                    [pixmap.width, pixmap.height],
                    pixmap.samples,
                )
                ocr_text = pytesseract.image_to_string(
                    image,
                    config="--psm 6",
                )
                text = (text + "\n" + ocr_text).strip()

            if text:
                pages_text.append(text)

    return "\n\n".join(pages_text).strip()


def extract_report_text(file_path):
    extension = Path(file_path).suffix.lower()

    if extension == ".pdf":
        return extract_from_pdf(file_path)

    if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return extract_from_image(file_path)

    raise ValueError("Unsupported file type. Upload a PDF or image.")


def detect_document_type(text, filename):
    content = f"{filename} {text}".lower()

    if any(word in content for word in ["prescription", "rx:", "dosage"]):
        return "Prescription"
    if any(word in content for word in ["discharge summary", "discharge date"]):
        return "Discharge Summary"
    if any(word in content for word in ["radiology", "impression", "x-ray", "mri"]):
        return "Imaging Report"
    if any(word in content for word in ["laboratory", "haemoglobin", "hemoglobin", "reference range"]):
        return "Laboratory Report"

    return "Medical Report"


def detect_report_date(text):
    patterns = [
        r"\b(\d{4}[-/]\d{1,2}[-/]\d{1,2})\b",
        r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)

    return "Not detected"