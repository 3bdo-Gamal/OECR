import os
import sys
from pathlib import Path
from typing import List, Optional
from pdf2image import convert_from_path
from PyPDF2 import PdfReader, PdfWriter
from core.config import get_setting

def get_poppler_path() -> Optional[str]:
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'poppler', 'bin')
    else:
        return None

def parse_page_ranges(range_str: str, max_pages: int) -> List[int]:
    """Parses a string like '1, 3, 5-7' into a sorted list of unique 0-indexed page numbers."""
    if not range_str or not range_str.strip():
        # Default to all pages
        return list(range(max_pages))
        
    pages = set()
    parts = range_str.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                start = max(1, start)
                end = min(max_pages, end)
                if start <= end:
                    pages.update(range(start - 1, end))
            except ValueError:
                continue # ignore invalid ranges
        else:
            try:
                page = int(part)
                if 1 <= page <= max_pages:
                    pages.add(page - 1)
            except ValueError:
                continue
                
    return sorted(list(pages)) if pages else list(range(max_pages))

def extract_pdf_pages(input_pdf_path: str, output_pdf_path: str, selected_pages: List[int]) -> str:
    reader = PdfReader(input_pdf_path)
    writer = PdfWriter()
    
    for page_num in selected_pages:
        if 0 <= page_num < len(reader.pages):
            writer.add_page(reader.pages[page_num])
            
    with open(output_pdf_path, "wb") as output_file:
        writer.write(output_file)
    return output_pdf_path

def convert_to_images(file_path: str, pages_str: str = "", output_folder: str = "temp_images") -> List[str]:
    file_path = Path(file_path)
    output_dir = Path(output_folder)
    output_dir.mkdir(exist_ok=True)
    image_paths = []
    
    supported_image_extensions = [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"]
    if file_path.suffix.lower() in supported_image_extensions:
        image_paths.append(str(file_path))
        return image_paths
        
    if file_path.suffix.lower() == ".pdf":
        reader = PdfReader(str(file_path))
        max_pages = len(reader.pages)
        selected_pages = parse_page_ranges(pages_str, max_pages)
        
        if not selected_pages:
            return []

        temp_pdf_path = output_dir / f"temp_{file_path.name}"
        extract_pdf_pages(str(file_path), str(temp_pdf_path), selected_pages)
        
        try:
            dpi = get_setting("ocr_dpi", 300)
            pages = convert_from_path(
                str(temp_pdf_path), 
                dpi=dpi, 
                poppler_path=get_poppler_path()
            ) 
            for i, page in enumerate(pages):
                # Use the actual page number for the filename (1-indexed)
                real_page_num = selected_pages[i] + 1
                image_name = output_dir / f"page_{real_page_num}.png"
                page.save(image_name, "PNG")
                image_paths.append(str(image_name))
        finally:
            if temp_pdf_path.exists():
                os.remove(temp_pdf_path)
                
        return image_paths
        
    raise ValueError("Unsupported file format.")