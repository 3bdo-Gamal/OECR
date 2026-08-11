import os
import cv2
import numpy as np
import logging
import threading
import re
from pathlib import Path
from core.config import get_setting
from typing import Dict, Any, List

# Suppress PaddleOCR logs globally
logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("paddle").setLevel(logging.ERROR)
# Suppress paddle C++ logs
os.environ["GLOG_minloglevel"] = "2"

class PaddleEngine:
    """Singleton pattern to ensure PaddleOCR models are loaded only once, unless settings change.
    
    When language is set to 'ar', two models are loaded (Arabic + English) 
    to support hybrid documents. Results from both models are merged.
    """
    _instance = None
    _ar_model = None
    _en_model = None
    _current_lang = None
    _current_gpu = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PaddleEngine, cls).__new__(cls)
        return cls._instance

    @classmethod
    def _set_device(cls, use_gpu: bool):
        """Attempt to set the paddle device."""
        try:
            import paddle  # type: ignore
            if use_gpu:
                paddle.device.set_device('gpu')
            else:
                paddle.device.set_device('cpu')
        except ImportError:
            pass
        except Exception:
            pass

    @classmethod
    def _initialize_model(cls):
        """Initializes PaddleOCR model(s) with optimized parameters."""
        from paddleocr import PaddleOCR
        
        use_gpu = get_setting("use_gpu", False)
        lang = get_setting("ocr_lang", "ar")
        
        cls._current_lang = lang
        cls._current_gpu = use_gpu

        cls._set_device(use_gpu)

        # Optimized parameters for Arabic + English accuracy
        common_params = dict(
            use_angle_cls=True,
            show_log=False,
            det_db_thresh=0.3,         # Text region detection threshold
            det_db_box_thresh=0.5,     # Box score threshold
            det_db_unclip_ratio=1.8,   # Expand detected text boxes for full capture
            det_limit_side_len=1920,   # Default 960 downscales 3x and loses detail
            rec_batch_num=6,           # Recognition batch size
            use_space_char=True,       # Better word spacing detection
            drop_score=0.3,            # Keep more raw detections, filter at our level
        )

        if lang == "ar":
            # Load both Arabic and English models for hybrid document support
            cls._ar_model = PaddleOCR(lang="ar", **common_params)
            cls._en_model = PaddleOCR(lang="en", **common_params)
        else:
            # English only
            cls._en_model = PaddleOCR(lang="en", **common_params)
            cls._ar_model = None

    def get_models(self) -> List:
        """Returns the loaded PaddleOCR model instances. Reinitializes if settings changed.
        
        Returns a list of (model, lang_code) tuples.
        """
        use_gpu = get_setting("use_gpu", False)
        lang = get_setting("ocr_lang", "ar")

        with self._lock:
            if (self._en_model is None or 
                use_gpu != self.__class__._current_gpu or 
                lang != self.__class__._current_lang):
                self.__class__._initialize_model()

        models = []
        if self.__class__._ar_model is not None:
            models.append((self.__class__._ar_model, "ar"))
        if self.__class__._en_model is not None:
            models.append((self.__class__._en_model, "en"))
        return models

def get_skew_angle(image: np.ndarray) -> float:
    """Calculates skew angle of the image using minAreaRect."""
    coords = np.column_stack(np.where(image > 0))
    if len(coords) == 0:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle

def deskew_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Deskews the image by the given angle."""
    if abs(angle) < 0.5:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _limit_image_size(img: np.ndarray, max_dim: int = 4096) -> np.ndarray:
    """Downscale image if it exceeds max dimension to prevent GPU OOM errors."""
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Adaptive preprocessing pipeline designed for Arabic + English OCR.
    
    Key design principles:
    - Arabic dots (نقط) are critical for letter distinction (ب/ت/ث/ن) —
      aggressive sharpening or filtering DESTROYS them
    - PaddleOCR has internal preprocessing, so we apply MINIMAL external processing
    - High-res images (from PDF scans) need less processing than low-res images
    - Uses fastNlMeansDenoising instead of bilateral filter (better for text)
    
    Returns grayscale image.
    """
    # 1. Grayscale conversion
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape[:2]
    
    # 2. Smart resize: limit max and upscale min
    if max(h, w) > 4096:
        scale = 4096 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    elif max(h, w) < 1000:
        scale = 1500 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    # 3. Denoise using Non-Local Means (far better for text than bilateral filter)
    #    h=10 is gentle enough to preserve Arabic dots while removing scan noise
    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)

    # 4. Adaptive CLAHE - only apply on low-contrast images
    contrast = denoised.std()
    if contrast < 60:
        # Low contrast — needs enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        denoised = clahe.apply(denoised)
    elif contrast < 80:
        # Medium contrast — gentle enhancement
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        denoised = clahe.apply(denoised)
    # High contrast images: skip CLAHE entirely, already good

    # 5. NO aggressive sharpening — it destroys Arabic dots and diacritics
    #    PaddleOCR's internal preprocessing handles this better

    # 6. Deskewing
    _, binary_for_skew = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    angle = get_skew_angle(binary_for_skew)
    if abs(angle) > 0.5:
        return deskew_image(denoised, angle)
    
    return denoised

def smart_sort_lines(boxes: List[Dict]) -> List[List[Dict]]:
    """Groups and sorts OCR results into visual lines by reading order.
    
    Returns a list of lines, where each line is a list of text items
    sorted by reading direction (RTL for Arabic, LTR for English).
    """
    if not boxes:
        return []

    # Calculate bounding box centers and heights
    def get_y_center_and_height(box_points):
        ys = [p[1] for p in box_points]
        return np.mean(ys), np.max(ys) - np.min(ys)
    
    boxes_with_info = []
    for item in boxes:
        box = item["bounding_box"]
        y_center, height = get_y_center_and_height(box)
        boxes_with_info.append({
            "item": item,
            "y_center": y_center,
            "height": height
        })
    
    # Sort by Y-center
    boxes_with_info.sort(key=lambda x: x["y_center"])

    # Group by horizontal band
    lines = []
    current_line = []
    current_y_center = None
    line_threshold = 0.65  # Max difference in y-center relative to line height
    
    for info in boxes_with_info:
        if current_y_center is None:
            current_y_center = info["y_center"]
            current_line.append(info)
            continue
            
        # Use max of current item height and average line height for better grouping
        avg_line_height = np.mean([item["height"] for item in current_line]) if current_line else 10
        threshold = line_threshold * max(info["height"], avg_line_height, 10)
        if abs(info["y_center"] - current_y_center) <= threshold:
            current_line.append(info)
            # Update running average of y_center
            current_y_center = np.mean([item["y_center"] for item in current_line])
        else:
            lines.append(current_line)
            current_line = [info]
            current_y_center = info["y_center"]
            
    if current_line:
        lines.append(current_line)
        
    sorted_lines = []
    lang = get_setting("ocr_lang", "ar")
    for line in lines:
        if lang == "ar":
            # Right to Left sort for Arabic
            line.sort(key=lambda info: np.max([p[0] for p in info["item"]["bounding_box"]]), reverse=True)
        else:
            # Left to Right sort for English
            line.sort(key=lambda info: np.min([p[0] for p in info["item"]["bounding_box"]]))
        sorted_lines.append([info["item"] for info in line])
        
    return sorted_lines

def _build_text_from_lines(line_groups: List[List[Dict]]) -> str:
    """Reconstruct properly formatted text from grouped line detection results.
    
    Intelligently joins text items within the same visual line with appropriate
    spacing based on bounding box gaps, and separates different lines with newlines.
    """
    lang = get_setting("ocr_lang", "ar")
    is_rtl = (lang == "ar")
    text_lines = []
    
    for line_items in line_groups:
        if not line_items:
            continue
        
        parts = []
        for i, item in enumerate(line_items):
            text = item["text"].strip()
            if not text:
                continue
            
            if i > 0 and parts:
                prev_item = line_items[i - 1]
                prev_box = prev_item["bounding_box"]
                curr_box = item["bounding_box"]
                
                # Calculate horizontal gap between consecutive boxes
                if is_rtl:
                    # RTL: previous box is to the right, current is to the left
                    gap = min(p[0] for p in prev_box) - max(p[0] for p in curr_box)
                else:
                    # LTR: previous box is to the left, current is to the right
                    gap = min(p[0] for p in curr_box) - max(p[0] for p in prev_box)
                
                # Estimate average character width for smart spacing decisions
                prev_width = max(p[0] for p in prev_box) - min(p[0] for p in prev_box)
                prev_chars = max(len(prev_item["text"].strip()), 1)
                avg_char_width = prev_width / prev_chars if prev_width > 0 else 10
                
                if gap > avg_char_width * 4:
                    parts.append("\t")  # Large gap → tab (tabular/columnar data)
                elif gap > avg_char_width * 0.3:
                    parts.append(" ")   # Normal gap → space between words
                # else: no space — boxes are adjacent/overlapping (same word split by OCR)
            
            parts.append(text)
        
        line_text = "".join(parts)
        if line_text.strip():
            text_lines.append(line_text)
    
    return "\n".join(text_lines)

def _postprocess_text(text: str) -> str:
    """Clean up common OCR artifacts and normalize formatting.
    
    Includes Arabic-specific fixes for broken words caused by PaddleOCR
    splitting connected Arabic text into multiple detection boxes.
    """
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # 1. Merge broken Arabic prefixes that PaddleOCR split off
        #    Only merge KNOWN Arabic prefixes: ال، و، ب، ف، ل، ك، لل، بال، وال، فال، كال
        #    e.g. "ال عتراف" -> "الاعتراف", "وال خسائر" -> "والخسائر"
        prefixes = r'(?:وال|بال|فال|كال|لل|ال|و|ب|ف|ل|ك)'
        pattern = r'(?:(?<=\s)|(?<=^))(' + prefixes + r') ([\u0600-\u06FF])'
        line = re.sub(pattern, r'\1\2', line)
        # Apply twice for cascading: "ال عتر اف" -> "الاعتر اف" -> handled by next pass
        line = re.sub(pattern, r'\1\2', line)
        
        # 2. Collapse multiple spaces to single space
        line = re.sub(r' {2,}', ' ', line)
        # 3. Normalize tab spacing
        line = re.sub(r'\t{2,}', '\t', line)
        stripped = line.strip()
        if stripped:
            cleaned.append(stripped)
    
    result = '\n'.join(cleaned)
    # Collapse 3+ consecutive newlines to double newline
    result = re.sub(r'\n{3,}', '\n\n', result)
    return result

def _overlap_ratio(box1, box2) -> float:
    """Calculate overlap ratio relative to the smaller box area.
    
    Unlike IoU, this catches cases where a small box is fully inside a larger one,
    which IoU would score low due to the large union area.
    """
    x1_min, y1_min = np.min(box1, axis=0)
    x1_max, y1_max = np.max(box1, axis=0)
    x2_min, y2_min = np.min(box2, axis=0)
    x2_max, y2_max = np.max(box2, axis=0)
    
    x_left = max(x1_min, x2_min)
    y_top = max(y1_min, y2_min)
    x_right = min(x1_max, x2_max)
    y_bottom = min(y1_max, y2_max)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = max((x1_max - x1_min) * (y1_max - y1_min), 1e-6)
    area2 = max((x2_max - x2_min) * (y2_max - y2_min), 1e-6)
    smaller_area = min(area1, area2)
    
    return intersection / smaller_area

def _is_arabic_text(text: str) -> bool:
    """Check if text is predominantly Arabic characters."""
    arabic_count = sum(1 for c in text if (
        '\u0600' <= c <= '\u06FF' or   # Arabic block (includes Arabic-Indic digits ٠-٩)
        '\u0750' <= c <= '\u077F' or   # Arabic Supplement
        '\uFB50' <= c <= '\uFDFF' or   # Arabic Presentation Forms-A
        '\uFE70' <= c <= '\uFEFF'      # Arabic Presentation Forms-B
    ))
    non_space = sum(1 for c in text if not c.isspace())
    if non_space == 0:
        return False
    return arabic_count / non_space > 0.3

def _fix_arabic_text(text: str) -> str:
    """Smart reversal for Arabic text from PaddleOCR.
    
    PaddleOCR's Arabic model outputs characters in reversed order (LTR instead of RTL).
    This function reverses the overall order while preserving LTR tokens like 
    numbers and English words in their correct reading direction.
    """
    if not _is_arabic_text(text):
        return text
    
    # Split into segments: LTR tokens (numbers, English) vs everything else
    segments = re.split(r'([A-Za-z0-9]+(?:[./][A-Za-z0-9]+)*)', text)
    
    # Reverse overall segment order (RTL), but keep LTR tokens intact
    reversed_segments = []
    for seg in reversed(segments):
        if re.match(r'^[A-Za-z0-9]+(?:[./][A-Za-z0-9]+)*$', seg):
            reversed_segments.append(seg)  # Keep numbers/English as-is
        else:
            reversed_segments.append(seg[::-1])  # Reverse Arabic segments
    
    return ''.join(reversed_segments)

def _parse_ocr_results(results, lang_code: str = "en") -> List[Dict]:
    """Parse PaddleOCR result list into structured items with confidence filtering."""
    items = []
    for line in results:
        if line is None:
            continue
        box = line[0]
        text = line[1][0]
        conf = float(line[1][1])
        
        # Fix Arabic text reversal from PaddleOCR
        if lang_code == "ar":
            text = _fix_arabic_text(text)
        
        # Filter by confidence (drop_score=0.3 lets more through from PaddleOCR)
        if conf >= 0.55: 
            items.append({"bounding_box": box, "text": text, "confidence": conf})
    return items

def _run_ocr_pass(model, img: np.ndarray, lang_code: str) -> List[Dict]:
    """Run OCR on an image with error recovery for GPU memory issues."""
    try:
        res = model.ocr(img, cls=True)
        results = res[0] if res and res[0] else []
        return _parse_ocr_results(results, lang_code)
    except Exception:
        # GPU OOM or other error — try with a smaller image
        h, w = img.shape[:2]
        if max(h, w) > 1500:
            scale = 1500 / max(h, w)
            smaller = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            try:
                res = model.ocr(smaller, cls=True)
                results = res[0] if res and res[0] else []
                return _parse_ocr_results(results, lang_code)
            except Exception:
                return []
        return []

def extract_text_from_image(image_path: str) -> Dict[str, Any]:
    try:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at path: {image_path}")
            
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise ValueError(f"Could not read the image: {image_path}. File might be corrupted or unsupported.")
        
        # Limit image size to prevent GPU memory errors
        orig_img = _limit_image_size(orig_img, max_dim=4096)
        
        # Apply preprocessing pipeline
        processed_img = preprocess_image(orig_img)
        processed_color = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2BGR)
        
        engine = PaddleEngine()
        models = engine.get_models()
        
        # ---- Dual-Pass Language-aware OCR Strategy ----
        # 1. Run OCR on BOTH original and preprocessed images
        # 2. Compare average confidence per model
        # 3. Pick the higher-confidence result set for each language
        # This handles cases where preprocessing helps OR hurts recognition.
        
        ar_items = []
        en_items = []
        
        for model, lang_code in models:
            # Pass 1: Preprocessed image (enhanced for OCR)
            items_proc = _run_ocr_pass(model, processed_color, lang_code)
            
            # Pass 2: Original image (PaddleOCR uses its own internal preprocessing)
            items_orig = _run_ocr_pass(model, orig_img, lang_code)
            
            # Pick the set with higher average confidence
            avg_conf_proc = np.mean([i["confidence"] for i in items_proc]) if items_proc else 0
            avg_conf_orig = np.mean([i["confidence"] for i in items_orig]) if items_orig else 0
            
            # Also consider detection count — prefer the set that found more text
            # (weighted: confidence matters more, but finding more text is a tiebreaker)
            score_proc = avg_conf_proc * (1 + 0.1 * len(items_proc))
            score_orig = avg_conf_orig * (1 + 0.1 * len(items_orig))
            
            best_items = items_proc if score_proc >= score_orig else items_orig
            
            if lang_code == "ar":
                ar_items = best_items
            else:
                en_items = best_items
        
        # --- Language-aware filtering ---
        # From Arabic model: keep only detections that contain Arabic text
        ar_filtered = [item for item in ar_items if _is_arabic_text(item["text"])]
        
        # From English model: keep only detections that do NOT contain Arabic text
        en_filtered = [item for item in en_items if not _is_arabic_text(item["text"])]
        
        # --- Merge: deduplicate overlapping regions ---
        all_items = list(ar_filtered)
        
        for en_item in en_filtered:
            has_overlap = False
            for ar_item in ar_filtered:
                if _overlap_ratio(en_item["bounding_box"], ar_item["bounding_box"]) > 0.2:
                    has_overlap = True
                    break
            if not has_overlap:
                all_items.append(en_item)
        
        # Smart line ordering - returns grouped lines
        line_groups = smart_sort_lines(all_items)
        
        # Flatten for layout data
        ordered_boxes = [item for line in line_groups for item in line]
        
        # Build properly formatted text with intelligent spacing
        full_text = _build_text_from_lines(line_groups)
        full_text = _postprocess_text(full_text)
        
        return {"raw_text": full_text.strip(), "layout_data": ordered_boxes}
        
    except Exception as e:
        raise Exception(f"OCR Engine Error: {str(e)}")