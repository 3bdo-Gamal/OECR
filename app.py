import customtkinter as ctk
from core.pdf_handler import get_poppler_path
from tkinter import filedialog
from pathlib import Path
from PIL import Image
from pdf2image import convert_from_path
from PyPDF2 import PdfReader

# Set the overall theme
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class DocumentProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure the window settings (Made wider for side-by-side layout)
        self.title("OECR")
        self.geometry("850x550")
        self.selected_file_path = None

        # --- ADD THE VARIABLES HERE ---
        self.current_preview_page = 1
        self.total_pdf_pages = 1

        # --- GRID LAYOUT ---
        self.grid_columnconfigure(0, weight=1) # Left side (Controls)
        self.grid_columnconfigure(1, weight=1) # Right side (Preview)
        self.grid_rowconfigure(0, weight=1)

        # ==========================================
        # LEFT FRAME: CONTROLS
        # ==========================================
        self.left_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.left_frame.grid(row=0, column=0, padx=20, pady=30, sticky="nsew")

        # Title Label
        self.title_label = ctk.CTkLabel(self.left_frame, text="OECR Processing", font=ctk.CTkFont(size=24, weight="bold"))
        self.title_label.pack(pady=(0, 20))

        # File Selection Button
        self.select_btn = ctk.CTkButton(self.left_frame, text="Select PDF or Image", command=self.browse_file)
        self.select_btn.pack(pady=10)

        # Label to show the path of the selected file
        self.file_label = ctk.CTkLabel(self.left_frame, text="No file selected", text_color="gray", wraplength=350)
        self.file_label.pack(pady=5)

        # Options Frame (Toggles for counting lines/words)
        self.options_frame = ctk.CTkFrame(self.left_frame)
        self.options_frame.pack(pady=30, fill="x")

        self.count_lines_var = ctk.StringVar(value="on")
        self.lines_checkbox = ctk.CTkCheckBox(self.options_frame, text="Count Lines", variable=self.count_lines_var, onvalue="on", offvalue="off")
        self.lines_checkbox.pack(side="left", padx=20, pady=15)

        self.count_words_var = ctk.StringVar(value="off")
        self.words_checkbox = ctk.CTkCheckBox(self.options_frame, text="Count Words", variable=self.count_words_var, onvalue="on", offvalue="off")
        self.words_checkbox.pack(side="right", padx=20, pady=15)

        # Action Button (Process)
        self.process_btn = ctk.CTkButton(self.left_frame, text="Process Document", command=self.process_document, fg_color="green", hover_color="darkgreen", height=40)
        self.process_btn.pack(pady=20)

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

    def browse_file(self):
        """Opens a native file dialog to pick a PDF or an Image."""
        file_types = [("Supported Files", "*.pdf *.png *.jpg *.jpeg"), ("PDF Files", "*.pdf"), ("Images", "*.png *.jpg *.jpeg")]
        filename = filedialog.askopenfilename(title="Select a Document", filetypes=file_types)
        
        if filename:
            self.selected_file_path = filename
            self.file_label.configure(text=f"Selected: {Path(filename).name}", text_color="white")
            self.update_preview(filename)

    def prev_page(self):
        if self.current_preview_page > 1:
            self.current_preview_page -= 1
            self.update_preview(self.selected_file_path, self.current_preview_page)

    def next_page(self):
        if self.current_preview_page < self.total_pdf_pages:
            self.current_preview_page += 1
            self.update_preview(self.selected_file_path, self.current_preview_page)

    def update_preview(self, filepath, page_num=1):
        """Generates a visual preview of a specific page."""
        try:
            self.preview_label.configure(text="Loading page...")
            self.update() 
            
            if filepath.lower().endswith(".pdf"):
                # Quickly find total pages without rendering
                if page_num == 1: 
                    reader = PdfReader(filepath)
                    self.total_pdf_pages = len(reader.pages)
                    
                # Show navigation controls if it's a multi-page PDF
                if self.total_pdf_pages > 1:
                    self.nav_frame.pack(side="bottom", pady=10)
                    self.page_info_label.configure(text=f"Page {page_num} / {self.total_pdf_pages}")
                else:
                    self.nav_frame.pack_forget()

                # Render only the requested page using the smart Poppler path
                pages = convert_from_path(
                    filepath, 
                    first_page=page_num, 
                    last_page=page_num, 
                    dpi=72,
                    poppler_path=get_poppler_path()
                )
                img = pages[0]
            else:
                # If it's a normal image, hide navigation
                self.nav_frame.pack_forget()
                img = Image.open(filepath)
            
            img.thumbnail((350, 450))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
            
            self.preview_label.configure(image=ctk_img, text="")
            self.preview_image_ref = ctk_img 
            
        except Exception as e:
            self.preview_label.configure(text="Preview not available", image="")
            print(f"Preview error: {e}")

    def process_document(self):
        """Placeholder function where our core OCR and text calculations will trigger."""
        if not self.selected_file_path:
            self.file_label.configure(text="Please select a file first!", text_color="red")
            return
        
        # Temporary status message
        print(f"Processing: {self.selected_file_path}")
        print(f"Count Lines: {self.count_lines_var.get()}, Count Words: {self.count_words_var.get()}")

# Start the desktop application loop
if __name__ == "__main__":
    app = DocumentProcessorApp()
    app.mainloop()