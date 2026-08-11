import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import hashlib
import json
import os
import threading
import shutil
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

try:
    from tkinterdnd2 import TkinterDnD, DND_FILES
    HAS_DND = True
except ImportError:
    HAS_DND = False

from core.pdf_handler import get_poppler_path, convert_to_images
from core.ocr_engine import extract_text_from_image
from core.text_analyzer import calculate_counts_from_raw_text
from core.document_builder import export_text_to_docx
from core.config import set_setting, get_setting

# Set the overall theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

if HAS_DND:
    class BaseApp(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    class BaseApp(ctk.CTk):
        pass

class DocumentProcessorApp(BaseApp):
    """Main Application Window for OECR."""
    
    def __init__(self) -> None:
        super().__init__()

        # Configure the window settings
        self.title("OECR")
        self.geometry("950x650")
        self.minsize(900, 600)
        self.selected_file_path = None

        # --- ADD THE VARIABLES HERE ---
        self.current_preview_page = 1
        self.total_pdf_pages = 1

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=1) # Left side (Controls)
        self.grid_columnconfigure(1, weight=1) # Right side (Preview)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT FRAME: CONTROLS (Sidebar)
        # ==========================================
        self.left_frame = ctk.CTkFrame(self, fg_color=("gray85", "gray17"), corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)
        self.left_frame.pack_propagate(False)
        
        # Add internal padding frame for better alignment
        self.sidebar_content = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.sidebar_content.pack(expand=True, fill="both", padx=20, pady=30)

        # Top Buttons Frame (Settings and Text Tools)
        self.top_buttons_frame = ctk.CTkFrame(self.sidebar_content, fg_color="transparent")
        self.top_buttons_frame.pack(anchor="ne", pady=(0, 10), fill="x")

        self.settings_btn = ctk.CTkButton(self.top_buttons_frame, text="⚙ Settings", width=60, command=self.open_settings)
        self.settings_btn.pack(side="right", padx=(5, 0))

        self.text_tools_btn = ctk.CTkButton(self.top_buttons_frame, text="📝 Text Tools", width=80, command=self.open_text_tools)
        self.text_tools_btn.pack(side="right")

        # Title Label
        self.title_label = ctk.CTkLabel(self.sidebar_content, text="OECR Processing", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(0, 20))

        # File Selection Button
        self.select_btn = ctk.CTkButton(self.sidebar_content, text="📁 Select PDF or Image", command=self.browse_file)
        self.select_btn.pack(pady=10)

        # Label to show the path of the selected file
        self.file_label = ctk.CTkLabel(self.sidebar_content, text="No file selected\n(Drag & Drop supported)" if HAS_DND else "No file selected", text_color=("black", "white"), wraplength=300)
        self.file_label.pack(pady=5)

        # Options Frame (Page Range for PDFs)
        self.options_frame = ctk.CTkFrame(self.sidebar_content)
        self.options_frame.pack(pady=30, fill="x")

        self.page_range_label = ctk.CTkLabel(self.options_frame, text="Pages (PDFs only):", font=ctk.CTkFont(weight="bold"))
        self.page_range_label.pack(pady=(10, 5))

        self.pages_var = ctk.StringVar()

        self.range_inputs_frame = ctk.CTkFrame(self.options_frame, fg_color="transparent")
        self.range_inputs_frame.pack(pady=(0, 15))

        self.pages_entry = ctk.CTkEntry(self.range_inputs_frame, textvariable=self.pages_var, width=150, placeholder_text="e.g. 1, 3, 5-7", state="disabled")
        self.pages_entry.pack(side="left", padx=5)

        # Action Button (Process)
        self.process_btn = ctk.CTkButton(self.sidebar_content, text="▶ Process Document", command=self.process_document, fg_color="green", hover_color="darkgreen", height=40)
        self.process_btn.pack(pady=20)

        # Progress Bar (Hidden by default)
        self.progress_bar = ctk.CTkProgressBar(self.sidebar_content, mode="determinate", width=250)
        self.progress_bar.set(0)

        # Status Label
        self.status_label = ctk.CTkLabel(self.sidebar_content, text="Ready", text_color="gray", font=ctk.CTkFont(size=12, slant="italic"))
        self.status_label.pack(side="bottom", pady=10)

        # ==========================================
        # RIGHT FRAME: DOCUMENT PREVIEW
        # ==========================================
        self.right_frame = ctk.CTkFrame(self)
        self.right_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.right_frame.pack_propagate(False) 
        
        # Label that will hold the image
        self.preview_label = ctk.CTkLabel(self.right_frame, text="Document Preview\nWill Appear Here", text_color="gray")
        self.preview_label.pack(expand=True, fill="both", pady=(10, 0))
        
        self.preview_image_ref = None 

        # --- Navigation Controls (Hidden by default) ---
        self.nav_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        
        self.prev_btn = ctk.CTkButton(self.nav_frame, text="< Prev", width=60, command=self.prev_page)
        self.prev_btn.pack(side="left", padx=10)
        
        self.page_info_label = ctk.CTkLabel(self.nav_frame, text="Page 1 / 1")
        self.page_info_label.pack(side="left", padx=10)
        
        self.next_btn = ctk.CTkButton(self.nav_frame, text="Next >", width=60, command=self.next_page)
        self.next_btn.pack(side="left", padx=10)

        # Drag & Drop Setup
        if HAS_DND:
            self.drop_target_register(DND_FILES)
            self.dnd_bind('<<Drop>>', self.handle_drop)

    def update_status(self, message: str) -> None:
        """Updates the status label."""
        self.status_label.configure(text=message)

    def handle_drop(self, event) -> None:
        """Handles drag & drop events."""
        file_path = event.data
        if file_path.startswith('{') and file_path.endswith('}'):
            file_path = file_path[1:-1]
            
        valid_exts = ['.pdf', '.png', '.jpg', '.jpeg']
        if any(file_path.lower().endswith(ext) for ext in valid_exts):
            self.selected_file_path = file_path
            self.file_label.configure(text=f"Selected: {Path(file_path).name}", text_color=("black", "white"))
            self.update_preview(file_path)
            self.update_status(f"Loaded {Path(file_path).name} via Drag & Drop")
        else:
            self.update_status("Unsupported file type dropped.")

    def open_text_tools(self) -> None:
        """Opens a window with text tools (live word, line, and character counter)."""
        tools_win = ctk.CTkToplevel(self)
        tools_win.title("Text Tools")
        tools_win.geometry("600x500")
        tools_win.attributes("-topmost", True)

        ctk.CTkLabel(tools_win, text="Text Tools", font=ctk.CTkFont(weight="bold", size=18)).pack(pady=15)

        text_box = ctk.CTkTextbox(tools_win, width=550, height=350)
        text_box.pack(pady=10)

        stats_frame = ctk.CTkFrame(tools_win, fg_color="transparent")
        stats_frame.pack(pady=10)

        words_label = ctk.CTkLabel(stats_frame, text="Words: 0", font=ctk.CTkFont(size=14))
        words_label.pack(side="left", padx=15)

        lines_label = ctk.CTkLabel(stats_frame, text="Lines: 0", font=ctk.CTkFont(size=14))
        lines_label.pack(side="left", padx=15)

        chars_label = ctk.CTkLabel(stats_frame, text="Characters: 0", font=ctk.CTkFont(size=14))
        chars_label.pack(side="left", padx=15)

        def update_counts(event=None):
            text = text_box.get("0.0", "end-1c")
            words = len(text.split())
            lines = text.count('\n') + 1 if text else 0
            chars = len(text)
            
            words_label.configure(text=f"Words: {words}")
            lines_label.configure(text=f"Lines: {lines}")
            chars_label.configure(text=f"Characters: {chars}")

        def update_counts_delayed(event=None):
            """Delayed update to allow tkinter to process paste/cut before counting."""
            tools_win.after(50, update_counts)

        text_box.bind("<KeyRelease>", update_counts)
        text_box.bind("<<Paste>>", update_counts_delayed)
        text_box.bind("<<Cut>>", update_counts_delayed)
        text_box.bind("<ButtonRelease-1>", update_counts)

    def open_settings(self) -> None:
        """Opens a small window to configure OCR settings."""
        settings_win = ctk.CTkToplevel(self)
        settings_win.title("Settings")
        settings_win.geometry("450x350")
        settings_win.attributes("-topmost", True)
        
        ctk.CTkLabel(settings_win, text="OECR Configuration", font=ctk.CTkFont(weight="bold", size=18)).pack(pady=15)
        
        # GPU Toggle
        use_gpu = get_setting("use_gpu", False)
        gpu_var = ctk.BooleanVar(value=use_gpu)
        
        def toggle_gpu():
            set_setting("use_gpu", gpu_var.get())
            
        gpu_switch = ctk.CTkSwitch(settings_win, text="Enable GPU Acceleration", variable=gpu_var, command=toggle_gpu)
        gpu_switch.pack(pady=10)

        # Language Selection
        lang = get_setting("ocr_lang", "ar")
        lang_var = ctk.StringVar(value="Arabic & English (Auto)" if lang == "ar" else "English Only")
        
        def change_lang(choice):
            lang_code = "ar" if choice == "Arabic & English (Auto)" else "en"
            set_setting("ocr_lang", lang_code)

        ctk.CTkLabel(settings_win, text="Primary Language:").pack(pady=(5, 0))
        lang_menu = ctk.CTkOptionMenu(settings_win, values=["Arabic & English (Auto)", "English Only"], command=change_lang)
        lang_menu.set(lang_var.get())
        lang_menu.pack(pady=5)

        # DPI Slider
        ctk.CTkLabel(settings_win, text="OCR DPI Quality:").pack(pady=(5, 0))
        dpi_var = ctk.IntVar(value=get_setting("ocr_dpi", 300))
        
        def dpi_changed(val):
            dpi_val = int(float(val))
            if dpi_val < 175:
                snap = 150
            elif dpi_val < 250:
                snap = 200
            else:
                snap = 300
            dpi_var.set(snap)
            set_setting("ocr_dpi", snap)
            dpi_label.configure(text=f"{snap} DPI")

        dpi_slider = ctk.CTkSlider(settings_win, from_=150, to=300, number_of_steps=2, command=dpi_changed)
        dpi_slider.set(dpi_var.get())
        dpi_slider.pack(pady=5)
        
        dpi_label = ctk.CTkLabel(settings_win, text=f"{dpi_var.get()} DPI")
        dpi_label.pack(pady=(0, 5))

        # Appearance Mode
        def change_appearance_mode(new_appearance_mode: str):
            ctk.set_appearance_mode(new_appearance_mode)
        
        ctk.CTkLabel(settings_win, text="Appearance Mode:").pack(pady=(5, 0))
        appearance_menu = ctk.CTkOptionMenu(settings_win, values=["System", "Light", "Dark"], command=change_appearance_mode)
        appearance_menu.set(ctk.get_appearance_mode())
        appearance_menu.pack(pady=5)

        # Clear Cache Button
        def clear_cache():
            cache_dir = Path("cache")
            if cache_dir.exists():
                try:
                    shutil.rmtree(cache_dir)
                    self.update_status("Cache cleared successfully!")
                except Exception as e:
                    self.update_status(f"Error clearing cache: {e}")
            else:
                self.update_status("Cache is already empty.")

        clear_cache_btn = ctk.CTkButton(settings_win, text="Clear Cache", command=clear_cache, fg_color="#ab3a3a", hover_color="#802b2b")
        clear_cache_btn.pack(pady=10)

    def browse_file(self) -> None:
        """Opens a native file dialog to pick a PDF or an Image."""
        file_types = [("Supported Files", "*.pdf *.png *.jpg *.jpeg"), ("PDF Files", "*.pdf"), ("Images", "*.png *.jpg *.jpeg")]
        filename = filedialog.askopenfilename(title="Select a Document", filetypes=file_types)
        
        if filename:
            self.selected_file_path = filename
            self.file_label.configure(text=f"Selected: {Path(filename).name}", text_color=("black", "white"))
            self.update_status(f"Selected {Path(filename).name}")
            self.update_preview(filename)

    def prev_page(self) -> None:
        if self.current_preview_page > 1:
            self.current_preview_page -= 1
            self.update_preview(self.selected_file_path, self.current_preview_page)

    def next_page(self) -> None:
        if self.current_preview_page < self.total_pdf_pages:
            self.current_preview_page += 1
            self.update_preview(self.selected_file_path, self.current_preview_page)

    def update_preview(self, filepath: str, page_num: int = 1) -> None:
        try:
            self.preview_label.configure(text="Loading page...")
            self.update() 
            
            if filepath.lower().endswith(".pdf"):
                if page_num == 1: 
                    reader = PdfReader(filepath)
                    self.total_pdf_pages = len(reader.pages)
                    self.pages_var.set(f"1-{self.total_pdf_pages}")
                    
                self.pages_entry.configure(state="normal")
                    
                if self.total_pdf_pages > 1:
                    self.nav_frame.pack(side="bottom", pady=10)
                    self.page_info_label.configure(text=f"Page {page_num} / {self.total_pdf_pages}")
                else:
                    self.nav_frame.pack_forget()

                pages = convert_from_path(
                    filepath, 
                    first_page=page_num, 
                    last_page=page_num, 
                    dpi=72,
                    poppler_path=get_poppler_path()
                )
                img = pages[0]
            else:
                self.nav_frame.pack_forget()
                self.pages_entry.configure(state="disabled")
                img = Image.open(filepath)
            
            img.thumbnail((350, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            
            self.preview_label.configure(image=ctk_img, text="")
            self.preview_image_ref = ctk_img 
            
        except Exception as e:
            self.preview_label.configure(text="Preview not available", image="")
            self.update_status(f"Preview error: {e}")

    def process_document(self) -> None:
        if not self.selected_file_path:
            self.file_label.configure(text="Please select a file first!", text_color="red")
            self.update_status("Error: No file selected")
            return
        
        self.process_btn.configure(state="disabled", text="Processing... Please Wait")
        self.progress_bar.set(0)
        self.progress_bar.pack(pady=10)
        self.update_status("Starting processing pipeline...")
        
        threading.Thread(target=self._run_processing_pipeline, daemon=True).start()

    def _get_file_hash(self, filepath: str, pages_str: str) -> str:
        # Pipeline version: bump this when preprocessing or engine params change
        # to automatically invalidate old cached results
        PIPELINE_VERSION = "v5"
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        settings_str = f"{PIPELINE_VERSION}_{pages_str}_{get_setting('ocr_lang', 'ar')}_{get_setting('use_gpu', False)}_{get_setting('ocr_dpi', 300)}"
        hasher.update(settings_str.encode('utf-8'))
        return hasher.hexdigest()

    def _run_processing_pipeline(self) -> None:
        try:
            is_pdf = self.selected_file_path.lower().endswith('.pdf')
            pages_str = self.pages_var.get() if is_pdf else ""

            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            
            file_hash = self._get_file_hash(self.selected_file_path, pages_str)
            cache_file = cache_dir / f"{file_hash}.json"

            full_extracted_text = ""

            if cache_file.exists():
                self.after(0, self.update_status, "Cache hit! Loading previous results...")
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    full_extracted_text = cached_data.get("raw_text", "")
                self.after(0, self.progress_bar.set, 1.0)
            else:
                self.after(0, self.update_status, "Converting document to images...")
                image_paths = convert_to_images(
                    file_path=self.selected_file_path,
                    pages_str=pages_str,
                    output_folder="temp_images"
                )
                
                total_pages = len(image_paths)
                self.after(0, self.update_status, f"Extracting text from {total_pages} pages using multiprocessing...")
                
                results_dict = {}
                with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                    future_to_index = {executor.submit(extract_text_from_image, img_path): i for i, img_path in enumerate(image_paths)}
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            ocr_result = future.result()
                            results_dict[idx] = ocr_result["raw_text"]
                        except Exception as exc:
                            results_dict[idx] = f"[Error extracting text from page {idx+1}: {exc}]"
                        
                        completed += 1
                        progress = completed / total_pages
                        self.after(0, self.progress_bar.set, progress)
                        self.after(0, self.update_status, f"Extracted text from {completed}/{total_pages} pages...")
                        
                for i in range(total_pages):
                    full_extracted_text += results_dict.get(i, "") + "\n\n"
                    
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"raw_text": full_extracted_text.strip()}, f, ensure_ascii=False, indent=4)

            self.after(0, self.show_results_window, full_extracted_text)
            self.after(0, self.update_status, "Processing complete.")

        except Exception as e:
            error_msg = f"An error occurred:\n{str(e)}"
            self.after(0, self.show_results_window, error_msg, True)
            self.after(0, self.update_status, "Processing failed.")
            
        finally:
            def reset_ui():
                self.process_btn.configure(state="normal", text="▶ Process Document")
                self.progress_bar.pack_forget()
            self.after(0, reset_ui)

    def show_results_window(self, text_data: str, is_error: bool = False) -> None:
        result_window = ctk.CTkToplevel(self)
        result_window.title("Processing Results")
        result_window.geometry("750x700")
        result_window.attributes("-topmost", True)

        if is_error:
            title = ctk.CTkLabel(result_window, text="Processing Failed", font=ctk.CTkFont(size=20, weight="bold"), text_color="red")
            title.pack(pady=20)
            
            error_box = ctk.CTkTextbox(result_window, width=700, height=500)
            error_box.pack(pady=10)
            error_box.insert("0.0", text_data)
            return

        title = ctk.CTkLabel(result_window, text="Processing Complete!", font=ctk.CTkFont(size=20, weight="bold"), text_color="green")
        title.pack(pady=(20, 10))

        counts_frame = ctk.CTkFrame(result_window, fg_color="transparent")
        counts_frame.pack(pady=10)

        counts_label = ctk.CTkLabel(counts_frame, text="", font=ctk.CTkFont(size=14, weight="bold"))
        
        def calculate_counts():
            counts = calculate_counts_from_raw_text(text_data, count_lines=True, count_words=True)
            counts_label.configure(text=f"Total Lines: {counts['lines']}  |  Total Words: {counts['words']}")
            counts_label.pack(pady=10)

        calc_btn = ctk.CTkButton(counts_frame, text="Calculate Lines & Words", command=calculate_counts)
        calc_btn.pack()

        ctk.CTkLabel(result_window, text="Extracted Text:", font=ctk.CTkFont(weight="bold")).pack(pady=(10, 5))
        text_box = ctk.CTkTextbox(result_window, width=700, height=400)
        text_box.pack(pady=5)
        text_box.insert("0.0", text_data)
        
        export_frame = ctk.CTkFrame(result_window, fg_color="transparent")
        export_frame.pack(pady=(10, 10))

        def on_export_docx():
            save_path = filedialog.asksaveasfilename(defaultextension=".docx", filetypes=[("Word Document", "*.docx")], title="Save Extracted Text as DOCX")
            if save_path:
                try:
                    success = export_text_to_docx(text_data, save_path)
                    if success:
                        self.update_status(f"Successfully saved to {Path(save_path).name}")
                    else:
                        self.update_status(f"Failed to save {Path(save_path).name}")
                except Exception as e:
                    self.update_status(f"Export error: {e}")

        def on_export_txt():
            save_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text File", "*.txt")], title="Save Extracted Text as TXT")
            if save_path:
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(text_data)
                    self.update_status(f"Successfully saved to {Path(save_path).name}")
                except Exception as e:
                    self.update_status(f"Export error: {e}")

        def on_export_json():
            save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON File", "*.json")], title="Save Extracted Text as JSON")
            if save_path:
                try:
                    with open(save_path, "w", encoding="utf-8") as f:
                        json.dump({"raw_text": text_data}, f, ensure_ascii=False, indent=4)
                    self.update_status(f"Successfully saved to {Path(save_path).name}")
                except Exception as e:
                    self.update_status(f"Export error: {e}")

        def on_copy_text():
            result_window.clipboard_clear()
            result_window.clipboard_append(text_data)
            self.update_status("Text copied to clipboard!")

        copy_btn = ctk.CTkButton(export_frame, text="📋 Copy Text", command=on_copy_text, fg_color="#186b51", hover_color="#124f3c")
        copy_btn.pack(side="left", padx=10)

        export_docx_btn = ctk.CTkButton(export_frame, text="📄 Export to .DOCX", command=on_export_docx, fg_color="#2b5b84", hover_color="#1f4363")
        export_docx_btn.pack(side="left", padx=10)
        
        export_txt_btn = ctk.CTkButton(export_frame, text="📝 Export to .TXT", command=on_export_txt, fg_color="#454545", hover_color="#2f2f2f")
        export_txt_btn.pack(side="left", padx=10)

        export_json_btn = ctk.CTkButton(export_frame, text="📦 Export to .JSON", command=on_export_json, fg_color="#916c27", hover_color="#6b4f1d")
        export_json_btn.pack(side="left", padx=10)

if __name__ == "__main__":
    app = DocumentProcessorApp()
    app.mainloop()