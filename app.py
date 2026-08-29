import sys
import os

# PyInstaller frozen mode: set working directory and paths
if getattr(sys, 'frozen', False):
    # Set working dir to exe's directory (for temp_images, cache, settings.json)
    os.chdir(os.path.dirname(sys.executable))
    # Point PaddleOCR to bundled models
    bundled_models = os.path.join(sys._MEIPASS, '.paddleocr')
    if os.path.exists(bundled_models):
        os.environ['PADDLEOCR_HOME'] = bundled_models

import customtkinter as ctk
from tkinter import filedialog
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import hashlib
import json
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
from core.ai_corrector import correct_text_with_ai, is_ai_correction_available

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
        settings_win.geometry("480x750")
        settings_win.attributes("-topmost", True)
        
        # Scrollable frame for all settings
        scroll_frame = ctk.CTkScrollableFrame(settings_win, width=440, height=570)
        scroll_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(scroll_frame, text="OECR Configuration", font=ctk.CTkFont(weight="bold", size=18)).pack(pady=(10, 15))
        
        # GPU Toggle
        use_gpu = get_setting("use_gpu", False)
        gpu_var = ctk.BooleanVar(value=use_gpu)
        
        def toggle_gpu():
            set_setting("use_gpu", gpu_var.get())
            
        gpu_switch = ctk.CTkSwitch(scroll_frame, text="Enable GPU Acceleration", variable=gpu_var, command=toggle_gpu)
        gpu_switch.pack(pady=8)

        # Language Selection
        lang = get_setting("ocr_lang", "ar")
        lang_var = ctk.StringVar(value="Arabic & English (Auto)" if lang == "ar" else "English Only")
        
        def change_lang(choice):
            lang_code = "ar" if choice == "Arabic & English (Auto)" else "en"
            set_setting("ocr_lang", lang_code)

        ctk.CTkLabel(scroll_frame, text="Primary Language:").pack(pady=(5, 0))
        lang_menu = ctk.CTkOptionMenu(scroll_frame, values=["Arabic & English (Auto)", "English Only"], command=change_lang)
        lang_menu.set(lang_var.get())
        lang_menu.pack(pady=5)

        # DPI Slider
        ctk.CTkLabel(scroll_frame, text="OCR DPI Quality:").pack(pady=(5, 0))
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

        dpi_slider = ctk.CTkSlider(scroll_frame, from_=150, to=300, number_of_steps=2, command=dpi_changed)
        dpi_slider.set(dpi_var.get())
        dpi_slider.pack(pady=5)
        
        dpi_label = ctk.CTkLabel(scroll_frame, text=f"{dpi_var.get()} DPI")
        dpi_label.pack(pady=(0, 5))

        # ---- OCR Engine Section ----
        separator_engine = ctk.CTkFrame(scroll_frame, height=2, fg_color="gray50")
        separator_engine.pack(fill="x", padx=20, pady=(10, 5))
        
        # Pre-declare ai_var so engine change can auto-toggle it
        ai_var = ctk.BooleanVar(value=get_setting("ai_correction", False))
        
        ctk.CTkLabel(scroll_frame, text="\U0001F50D OCR Engine", font=ctk.CTkFont(weight="bold", size=15)).pack(pady=(5, 2))
        
        # Engine Selection
        engine_map = {
            "Auto (Smart Routing)": "auto",
            "PaddleOCR (Local)": "paddle",
            "Gemini Vision (Cloud) ⚡ Fastest": "gemini_vision"
        }
        engine_reverse = {v: k for k, v in engine_map.items()}
        current_engine = get_setting("ocr_engine", "auto")
        
        engine_desc_texts = {
            "auto": "Local OCR first → falls back to Gemini if quality is low",
            "paddle": "Offline only, no internet needed. Slower, lower Arabic accuracy",
            "gemini_vision": "⚡ Fastest & most accurate. Sends images to Google Gemini API"
        }
        
        engine_desc = ctk.CTkLabel(scroll_frame, text=engine_desc_texts.get(current_engine, ""),
                                    font=ctk.CTkFont(size=11), text_color="gray60", wraplength=400)
        engine_desc.pack(pady=(0, 3))
        
        def change_engine(choice):
            engine_code = engine_map.get(choice, "auto")
            set_setting("ocr_engine", engine_code)
            engine_desc.configure(text=engine_desc_texts.get(engine_code, ""))
            
            # Auto-configure related settings
            if engine_code == "gemini_vision":
                # Gemini Vision produces high-quality text — AI correction is redundant
                ai_var.set(False)
                set_setting("ai_correction", False)
                cloud_var.set(True)
                set_setting("allow_cloud_ocr", True)
            elif engine_code == "auto":
                cloud_var.set(True)
                set_setting("allow_cloud_ocr", True)
        
        ctk.CTkLabel(scroll_frame, text="OCR Engine:").pack(pady=(5, 0))
        engine_menu = ctk.CTkOptionMenu(
            scroll_frame,
            values=list(engine_map.keys()),
            command=change_engine
        )
        engine_menu.set(engine_reverse.get(current_engine, "Auto (Smart Routing)"))
        engine_menu.pack(pady=5)
        
        # Cloud OCR Toggle
        cloud_enabled = get_setting("allow_cloud_ocr", True)
        cloud_var = ctk.BooleanVar(value=cloud_enabled)
        
        def toggle_cloud():
            set_setting("allow_cloud_ocr", cloud_var.get())
        
        cloud_switch = ctk.CTkSwitch(scroll_frame, text="Allow Cloud OCR (sends images to Gemini)", variable=cloud_var, command=toggle_cloud)
        cloud_switch.pack(pady=5)
        
        # Skip Tables Toggle
        skip_tables = get_setting("skip_tables", False)
        skip_var = ctk.BooleanVar(value=skip_tables)
        
        def toggle_skip_tables():
            set_setting("skip_tables", skip_var.get())
        
        skip_switch = ctk.CTkSwitch(scroll_frame, text="Skip Tables (text-only mode)", variable=skip_var, command=toggle_skip_tables)
        skip_switch.pack(pady=5)

        # ---- Gemini API Keys Section ----
        separator = ctk.CTkFrame(scroll_frame, height=2, fg_color="gray50")
        separator.pack(fill="x", padx=20, pady=(10, 5))
        
        ctk.CTkLabel(scroll_frame, text="🔑 Gemini API Keys", font=ctk.CTkFont(weight="bold", size=15)).pack(pady=(5, 2))
        ctk.CTkLabel(scroll_frame, text="Enter up to 5 keys (one per line). Auto-rotates when limit is reached.", 
                     font=ctk.CTkFont(size=11), text_color="gray60", wraplength=400).pack(pady=(0, 5))
        
        # Multi-key text area
        api_keys_textbox = ctk.CTkTextbox(scroll_frame, width=400, height=120, 
                                           font=ctk.CTkFont(size=12, family="Consolas"))
        existing_keys = get_setting("gemini_api_keys", [])
        if not existing_keys:
            # Backward compat: migrate single key
            single_key = get_setting("gemini_api_key", "")
            if single_key:
                existing_keys = [single_key]
        
        api_keys_textbox.insert("1.0", "\n".join(k for k in existing_keys if k))
        api_keys_textbox.pack(pady=5)
        
        key_status_label = ctk.CTkLabel(scroll_frame, text="", font=ctk.CTkFont(size=11))
        key_status_label.pack(pady=(0, 3))
        
        def save_api_keys(event=None):
            text = api_keys_textbox.get("1.0", "end").strip()
            keys = [k.strip() for k in text.split("\n") if k.strip()]
            # Pad to 5 slots
            while len(keys) < 5:
                keys.append("")
            keys = keys[:5]  # Max 5
            set_setting("gemini_api_keys", keys)
            # Also set single key for backward compat
            first_key = next((k for k in keys if k), "")
            set_setting("gemini_api_key", first_key)
            
            valid_count = sum(1 for k in keys if k)
            key_status_label.configure(
                text=f"{valid_count} key(s) configured",
                text_color="#51cf66" if valid_count > 0 else "#ff6b6b"
            )
            # Reset key manager exhaustion tracking when keys change
            try:
                from core.api_key_manager import APIKeyManager
                APIKeyManager().reset_all()
            except Exception:
                pass
        
        api_keys_textbox.bind("<FocusOut>", save_api_keys)
        
        # Show initial count
        valid_count = sum(1 for k in existing_keys if k)
        key_status_label.configure(
            text=f"{valid_count} key(s) configured",
            text_color="#51cf66" if valid_count > 0 else "#ff6b6b"
        )
        
        # Test All Keys Button
        def test_all_keys():
            save_api_keys()
            keys = [k for k in get_setting("gemini_api_keys", []) if k.strip()]
            if not keys:
                test_status.configure(text="❌ No API keys entered", text_color="#ff6b6b")
                return
            
            try:
                from google import genai
            except ImportError:
                test_status.configure(text="❌ google-genai not installed", text_color="#ff6b6b")
                return
            
            model_name = get_setting("gemini_model", "gemini-3.6-flash")
            results = []
            working = 0
            
            for i, key in enumerate(keys):
                try:
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(
                        model=model_name,
                        contents="Reply with only: OK"
                    )
                    if response and response.text:
                        results.append(f"Key {i+1}: ✅")
                        working += 1
                    else:
                        results.append(f"Key {i+1}: ⚠️ Empty response")
                except Exception as e:
                    err = str(e)[:40]
                    if "429" in str(e) or "quota" in str(e).lower():
                        results.append(f"Key {i+1}: ⚠️ Rate limited")
                    else:
                        results.append(f"Key {i+1}: ❌ {err}")
            
            summary = " | ".join(results)
            color = "#51cf66" if working == len(keys) else "#ffa94d" if working > 0 else "#ff6b6b"
            test_status.configure(text=f"{working}/{len(keys)} working — {summary}", text_color=color)
        
        test_btn = ctk.CTkButton(scroll_frame, text="Test All Keys", command=test_all_keys,
                                 width=150, fg_color="#2b8a3e", hover_color="#237032")
        test_btn.pack(pady=5)
        
        test_status = ctk.CTkLabel(scroll_frame, text="", font=ctk.CTkFont(size=11), wraplength=420)
        test_status.pack(pady=(0, 5))
        
        # ---- AI Correction Toggle ----
        ctk.CTkLabel(scroll_frame, text="🧠 AI Text Correction", font=ctk.CTkFont(weight="bold", size=13)).pack(pady=(10, 2))
        ctk.CTkLabel(scroll_frame, text="Extra Gemini pass to fix OCR errors (only for PaddleOCR)", 
                     font=ctk.CTkFont(size=11), text_color="gray60").pack(pady=(0, 3))
        
        # AI Toggle (ai_var declared above in engine section)
        ai_enabled = ai_var.get()
        
        def toggle_ai():
            set_setting("ai_correction", ai_var.get())
            
        ai_switch = ctk.CTkSwitch(scroll_frame, text="Enable AI Correction", variable=ai_var, command=toggle_ai)
        ai_switch.pack(pady=5)
        
        separator2 = ctk.CTkFrame(scroll_frame, height=2, fg_color="gray50")
        separator2.pack(fill="x", padx=20, pady=(5, 10))

        # Appearance Mode
        def change_appearance_mode(new_appearance_mode: str):
            ctk.set_appearance_mode(new_appearance_mode)
        
        ctk.CTkLabel(scroll_frame, text="Appearance Mode:").pack(pady=(5, 0))
        appearance_menu = ctk.CTkOptionMenu(scroll_frame, values=["System", "Light", "Dark"], command=change_appearance_mode)
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

        clear_cache_btn = ctk.CTkButton(scroll_frame, text="Clear Cache", command=clear_cache, fg_color="#ab3a3a", hover_color="#802b2b")
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
        PIPELINE_VERSION = "v9"
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        settings_str = (
            f"{PIPELINE_VERSION}_{pages_str}"
            f"_{get_setting('ocr_engine', 'auto')}"
            f"_{get_setting('ocr_lang', 'ar')}"
            f"_{get_setting('use_gpu', False)}"
            f"_{get_setting('ocr_dpi', 300)}"
            f"_{get_setting('skip_tables', False)}"
        )
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
            rate_limit_hit = False
            rate_limit_reset = ""

            if cache_file.exists():
                # Ask the user whether to use cached result or reprocess
                use_cache_event = threading.Event()
                use_cache_result = [True]  # default: use cache

                def ask_cache_choice():
                    dialog = ctk.CTkToplevel(self)
                    dialog.title("Cached Result Found")
                    dialog.geometry("420x200")
                    dialog.attributes("-topmost", True)
                    dialog.grab_set()
                    dialog.resizable(False, False)

                    ctk.CTkLabel(
                        dialog,
                        text="📋 This file was already processed.",
                        font=ctk.CTkFont(size=15, weight="bold")
                    ).pack(pady=(20, 5))
                    ctk.CTkLabel(
                        dialog,
                        text="Do you want to use the cached result or reprocess the file?",
                        font=ctk.CTkFont(size=12),
                        text_color="gray70",
                        wraplength=380
                    ).pack(pady=(0, 15))

                    btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
                    btn_frame.pack()

                    def use_cached():
                        use_cache_result[0] = True
                        dialog.destroy()
                        use_cache_event.set()

                    def reprocess():
                        use_cache_result[0] = False
                        dialog.destroy()
                        use_cache_event.set()

                    ctk.CTkButton(
                        btn_frame, text="✅ Use Cached Result",
                        command=use_cached, width=170,
                        fg_color="#2b8a3e", hover_color="#237032"
                    ).pack(side="left", padx=8)

                    ctk.CTkButton(
                        btn_frame, text="🔄 Reprocess",
                        command=reprocess, width=170,
                        fg_color="#1971c2", hover_color="#1864ab"
                    ).pack(side="left", padx=8)

                    dialog.protocol("WM_DELETE_WINDOW", use_cached)  # X button = use cache

                self.after(0, ask_cache_choice)
                use_cache_event.wait()  # Block background thread until user responds

                if use_cache_result[0]:
                    # Load from cache
                    self.after(0, self.update_status, "Loading cached result...")
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cached_data = json.load(f)
                        full_extracted_text = cached_data.get("raw_text", "")
                    self.after(0, self.progress_bar.set, 1.0)
                else:
                    # Delete old cache so pipeline overwrites it at the end
                    cache_file.unlink(missing_ok=True)
                    self.after(0, self.update_status, "Reprocessing file...")

            if not full_extracted_text:
                # Either no cache existed, or user chose to reprocess (cache was deleted above)
                self.after(0, self.update_status, "Converting document to images...")
                image_paths = convert_to_images(
                    file_path=self.selected_file_path,
                    pages_str=pages_str,
                    output_folder="temp_images"
                )
                
                total_pages = len(image_paths)
                engine_mode = get_setting('ocr_engine', 'auto')
                self.after(0, self.update_status, f"Extracting text from {total_pages} pages (engine: {engine_mode})...")
                
                # Track whether any page used Gemini Vision (to skip AI correction)
                used_gemini_vision = False
                results_dict = {}
                
                # Use the new OCR router for each page
                from core.ocr_router import extract_text as router_extract_text
                
                def process_single_page(img_path):
                    """Process a single page through the OCR router."""
                    result = router_extract_text(img_path)
                    return result
                
                # For Gemini Vision, use single thread to respect rate limits
                if engine_mode == 'gemini_vision':
                    max_w = 1
                else:
                    max_w = min(4, os.cpu_count() or 1)
                    
                # Track rate limit info
                rate_limit_hit = False
                rate_limit_reset = ""
                
                with ThreadPoolExecutor(max_workers=max_w) as executor:
                    future_to_index = {
                        executor.submit(process_single_page, img_path): i 
                        for i, img_path in enumerate(image_paths)
                    }
                    completed = 0
                    for future in concurrent.futures.as_completed(future_to_index):
                        idx = future_to_index[future]
                        try:
                            ocr_result = future.result()
                            results_dict[idx] = ocr_result.raw_text
                            if ocr_result.engine == 'gemini_vision':
                                used_gemini_vision = True
                            # Check if this page needs AI correction
                            if ocr_result.metadata.get('needs_ai_correction'):
                                results_dict[f"{idx}_needs_ai"] = True
                            # Track rate limit
                            if ocr_result.metadata.get('rate_limit'):
                                rate_limit_hit = True
                                rate_limit_reset = ocr_result.metadata.get('rate_limit_reset', '')
                        except Exception as exc:
                            results_dict[idx] = f"[Error extracting text from page {idx+1}: {exc}]"
                        
                        completed += 1
                        progress = completed / total_pages
                        self.after(0, self.progress_bar.set, progress)
                        status_msg = f"Extracted text from {completed}/{total_pages} pages..."
                        if rate_limit_hit:
                            reset_info = f" (API limit reached{', resets in ' + rate_limit_reset if rate_limit_reset else ''})"
                            status_msg += reset_info
                        self.after(0, self.update_status, status_msg)
                        
                for i in range(total_pages):
                    full_extracted_text += results_dict.get(i, "") + "\n\n"
                
                full_extracted_text = full_extracted_text.strip()
                
                # ---- AI Correction Pass ----
                # Only apply AI correction for PaddleOCR results, NOT for Gemini Vision
                # (Gemini Vision already produces high-quality text; double-correcting causes hallucinations)
                needs_ai = not used_gemini_vision and is_ai_correction_available()
                if needs_ai:
                    self.after(0, self.update_status, "Applying AI text correction (Gemini)...")
                    self.after(0, self.progress_bar.set, 0.0)
                    
                    def ai_progress(msg):
                        self.after(0, self.update_status, msg)
                    
                    corrected_text = correct_text_with_ai(full_extracted_text, progress_callback=ai_progress)
                    if corrected_text:
                        full_extracted_text = corrected_text
                    self.after(0, self.progress_bar.set, 1.0)
                elif used_gemini_vision:
                    self.after(0, self.update_status, "Gemini Vision used — skipping AI correction (already high quality).")
                    
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({"raw_text": full_extracted_text}, f, ensure_ascii=False, indent=4)

            self.after(0, self.show_results_window, full_extracted_text)
            if rate_limit_hit:
                reset_info = f" Resets in {rate_limit_reset}." if rate_limit_reset else ""
                self.after(0, self.update_status, f"Processing complete. ⚠️ API limit was reached.{reset_info}")
            else:
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
        
        # Check if text has Arabic content
        has_arabic = any('\u0600' <= c <= '\u06FF' for c in text_data)
        
        # RTL display toggle for Arabic text
        if has_arabic:
            rtl_frame = ctk.CTkFrame(result_window, fg_color="transparent")
            rtl_frame.pack(pady=(0, 5))
            
            rtl_var = ctk.BooleanVar(value=True)
            
            def make_rtl_text(text):
                """Reverse word order of Arabic lines so LTR widget displays them correctly."""
                lines = text.split('\n')
                fixed = []
                for line in lines:
                    line_has_ar = any('\u0600' <= c <= '\u06FF' for c in line)
                    if line_has_ar and line.strip():
                        # Reverse word order for display in LTR widget
                        words = line.split()
                        fixed.append(' '.join(reversed(words)))
                    else:
                        fixed.append(line)
                return '\n'.join(fixed)
            
            def toggle_rtl():
                inner = text_box._textbox
                inner.delete("1.0", "end")
                if rtl_var.get():
                    display = make_rtl_text(text_data)
                    inner.tag_configure("rtl", justify="right")
                    inner.insert("1.0", display)
                    inner.tag_add("rtl", "1.0", "end")
                else:
                    inner.insert("1.0", text_data)
            
            rtl_switch = ctk.CTkSwitch(rtl_frame, text="Fix Arabic Display (RTL)", 
                                        variable=rtl_var, command=toggle_rtl)
            rtl_switch.pack(side="left", padx=10)
            
            ctk.CTkLabel(rtl_frame, text="Toggle if Arabic text appears reversed",
                        font=ctk.CTkFont(size=11), text_color="gray60").pack(side="left")
        
        text_box = ctk.CTkTextbox(result_window, width=700, height=400)
        text_box.pack(pady=5)
        
        if has_arabic:
            # Default: show RTL-fixed display
            inner_text = text_box._textbox
            inner_text.tag_configure("rtl", justify="right")
            inner_text.insert("1.0", make_rtl_text(text_data))
            inner_text.tag_add("rtl", "1.0", "end")
        else:
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
                    with open(save_path, "w", encoding="utf-8-sig") as f:
                        # Add RTL mark for Arabic content
                        has_ar = any('\u0600' <= c <= '\u06FF' for c in text_data)
                        if has_ar:
                            f.write('\u200F')  # Right-to-Left Mark
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