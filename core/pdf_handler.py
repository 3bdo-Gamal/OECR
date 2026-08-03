import os
import sys
from pathlib import Path
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter

def get_poppler_path():
    """
    Dynamically routes Poppler. 
    If running as an .exe, it points to the bundled folder.
    If running as a script, it returns None (forcing it to use your System PATH).
    """
    if getattr(sys, 'frozen', False):
        # The app is running as a compiled .exe
        return os.path.join(sys._MEIPASS, 'poppler', 'bin')
    else:
        # The app is running as a normal Python script (uses Environment Variables)
        return None

def extract_pdf_pages(input_pdf_path, output_pdf_path, start_page=1, end_page=None):
    """
    Extracts a specific range of pages from a PDF and saves them as a new temporary PDF.
    """
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()

    total_pages = len(reader.pages)
    
    if end_page is None or end_page > total_pages:
        end_page = total_pages

    for page_num in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_num])

    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)

    return output_pdf_path

def convert_to_images(file_path, start_page=1, end_page=None, output_folder="temp_images"):
    """
    Takes a file (PDF or Image). If PDF, extracts the pages and converts them to images.
    Returns a list of image file paths.
    """
    file_path = Path(file_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    
    image_paths = []

    if file_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
        image_paths.append(str(file_path))
        return image_paths

    if file_path.suffix.lower() == ".pdf":
        temp_pdf_path = output_dir / f"temp_{file_path.name}"
        
        extract_pdf_pages(file_path, temp_pdf_path, start_page, end_page)
        
        # We pass the dynamic poppler path into the conversion function
        pages = convert_from_path(
            temp_pdf_path, 
            dpi=300, 
            poppler_path=get_poppler_path()
        ) 
        
        for i, page in enumerate(pages):
            image_name = output_dir / f"page_{start_page + i}.jpg"
            page.save(image_name, "JPEG")
            image_paths.append(str(image_name))
            
        os.remove(temp_pdf_path)
        
        return image_paths

    raise ValueError("Unsupported file format.")