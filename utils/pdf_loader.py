
import PyPDF2
from pathlib import Path

def load_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF file.
    Returns a single string with all page content.
    """
    text_parts = []

    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)

        # Loop through every page
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text()

            # Some pages may be empty (image-only pages)
            if page_text:
                # Add page number marker for traceability
                text_parts.append(f"\n--- Page {page_num+1} ---\n{page_text}")

    return "\n".join(text_parts)