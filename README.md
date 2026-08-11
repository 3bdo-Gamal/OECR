# OECR (Optical Extraction & Counting Routine)

OECR is a native Windows desktop application built with Python. Its primary purpose is to process documents (PDFs and images—including scanned and Arabic text), extract the text with maximum accuracy, calculate exact line and word counts, and export the extracted text into an editable Microsoft Word document.

## 🌟 Key Features
- **Dynamic File Preview:** Visually navigate through your PDFs and Images before processing them.
- **Page Range Selection:** Select precisely which pages of a multi-page PDF you want to process, saving time and API costs.
- **Table Exclusion:** Currently, this architecture processes text strictly by lines natively from PaddleOCR, simplifying the pipeline while maintaining accuracy.
- **On-Demand Text Analytics:** Calculate the total number of words and non-empty lines instantly after text is extracted.
- **Export to DOCX:** With a single click, export your extracted text into a structured Microsoft Word (.docx) file for editing.
- **Settings Management:** Configure your OCR engine preferences (e.g., Enable GPU, Language selection) directly through the app UI, saved persistently.

## 🛠️ Technology Stack
- **UI Framework:** `customtkinter` (Dark-themed modern GUI)
- **PDF Handling:** `PyPDF2`, `pdf2image`, `Pillow`
- **OCR Engine:** `PaddleOCR` (Local, fast, open-source text extraction powered by deep learning).
- **Image Processing:** `opencv-python` (Noise reduction, grayscale conversion, and adaptive thresholding).
- **Document Export:** `python-docx`
- **Rendering Engine:** `poppler` (dynamically routed for `.exe` bundling).

## 🚀 Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone <your-repo-url>
   cd OECR
   ```

2. **Create a Virtual Environment (Recommended):**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Poppler (For PDF rendering):**
   - Download the latest Poppler binaries for Windows.
   - Extract them and add the `poppler/bin` directory to your System PATH.

5. **Run the Application:**
   ```bash
   python app.py
   ```

## ⚙️ Configuration (PaddleOCR)
OECR uses PaddleOCR for its robust, offline OCR capabilities. 
- You can configure the engine directly from the application's **⚙ Settings** menu.
- **Enable GPU Acceleration:** If you have an NVIDIA GPU with CUDA installed, toggling this will significantly speed up processing.
- **Primary Language:** Select between English and Arabic.
*Note: Any changes to OCR settings require you to restart the application to take effect.*

## 📁 Project Structure
- `app.py`: The main GUI application and threading logic.
- `core/config.py`: Handles saving and loading user preferences (`settings.json`).
- `core/pdf_handler.py`: Manages PDF parsing, page extraction, and conversion to high-res images.
- `core/ocr_engine.py`: Preprocesses images with OpenCV and runs the PaddleOCR singleton engine.
- `core/text_analyzer.py`: Algorithms to calculate words and valid lines.
- `core/document_builder.py`: Generates the structured `.docx` files.
