import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

from core.ocr_base import OCREngine, OCRResult
from core.ocr_quality import analyze_ocr_quality, validate_financial_numbers
from core.ocr_gemini_vision import RateLimitError
from core.config import get_setting

logger = logging.getLogger(__name__)

def get_available_engines() -> Dict[str, OCREngine]:
    """Get all available OCR engines."""
    engines: Dict[str, OCREngine] = {}
    
    try:
        from core.ocr_paddle import PaddleOCREngine
        paddle = PaddleOCREngine()
        if paddle.is_available():
            engines[paddle.name] = paddle
    except ImportError as e:
        logger.warning(f"Failed to load PaddleOCREngine: {e}")
        
    try:
        from core.ocr_gemini_vision import GeminiVisionOCREngine
        gemini = GeminiVisionOCREngine()
        if gemini.is_available():
            engines[gemini.name] = gemini
    except ImportError as e:
        logger.warning(f"Failed to load GeminiVisionOCREngine: {e}")
        
    return engines

def get_engine(name: str) -> OCREngine:
    """Get a specific engine by name."""
    engines = get_available_engines()
    if name not in engines:
        raise ValueError(f"OCR Engine '{name}' is not available.")
    return engines[name]

def extract_text(image_path: str, 
                 engine_name: str = None,
                 progress_callback=None) -> OCRResult:
    """Main entry point for all OCR operations."""
    mode = engine_name or get_setting("ocr_engine", "auto")
    lang = get_setting("ocr_lang", "ar")
    allow_cloud = get_setting("allow_cloud_ocr", True)
    skip_tables = get_setting("skip_tables", False)
    quality_threshold = float(get_setting("quality_threshold", 0.65))
    ai_correction = get_setting("ai_correction", False)
    
    engines = get_available_engines()
    
    if mode == "auto":
        paddle_engine = engines.get("paddle")
        if not paddle_engine:
            if progress_callback:
                progress_callback("PaddleOCR unavailable in auto mode.")
            return _fallback_result("All engines failed")
            
        if progress_callback:
            progress_callback("Running PaddleOCR...")
            
        try:
            paddle_res = paddle_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
            if ai_correction:
                paddle_res.metadata['needs_ai_correction'] = True
                
            report = analyze_ocr_quality(paddle_res)
            
            if report.score >= quality_threshold:
                return paddle_res
            else:
                if "gemini_vision" in engines and allow_cloud:
                    if progress_callback:
                        progress_callback(f"Local OCR quality low (score={report.score:.2f}), trying Gemini Vision...")
                    
                    try:
                        gemini_engine = engines["gemini_vision"]
                        gemini_res = gemini_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
                        
                        validation = validate_financial_numbers(gemini_res.raw_text, paddle_res.raw_text)
                        gemini_res.metadata['financial_validation'] = validation
                        return gemini_res
                    
                    except RateLimitError as e:
                        logger.warning(f"Gemini Vision rate limit in auto fallback: {e}")
                        if progress_callback:
                            reset_info = f" Resets in {e.reset_time}." if e.reset_time else ""
                            progress_callback(f"⚠️ API limit reached.{reset_info} Using local OCR result.")
                        paddle_res.metadata['rate_limit'] = True
                        paddle_res.metadata['rate_limit_reset'] = e.reset_time
                        return paddle_res
                        
                    except Exception as e:
                        logger.error(f"Gemini Vision failed in auto fallback: {e}")
                        if progress_callback:
                            progress_callback("Gemini Vision failed, falling back to local OCR result.")
                        return paddle_res
                else:
                    return paddle_res
                    
        except Exception as e:
            logger.error(f"PaddleOCR failed in auto mode: {e}")
            return _fallback_result("All engines failed")
            
    elif mode == "paddle":
        if "paddle" not in engines:
            return _fallback_result("PaddleOCR is not available.")
            
        if progress_callback:
            progress_callback("Running PaddleOCR...")
            
        try:
            paddle_engine = engines["paddle"]
            res = paddle_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
            if ai_correction:
                res.metadata['needs_ai_correction'] = True
            return res
        except Exception as e:
            logger.error(f"PaddleOCR failed: {e}")
            return _fallback_result(f"PaddleOCR failed: {e}")
            
    elif mode == "gemini_vision":
        if "gemini_vision" not in engines or not allow_cloud:
            if progress_callback:
                progress_callback("Gemini Vision unavailable or not allowed, trying PaddleOCR...")
            if "paddle" in engines:
                try:
                    paddle_engine = engines["paddle"]
                    res = paddle_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
                    if ai_correction:
                        res.metadata['needs_ai_correction'] = True
                    return res
                except Exception as e:
                    logger.error(f"Fallback to PaddleOCR failed: {e}")
                    return _fallback_result(f"All engines failed: {e}")
            return _fallback_result("Gemini Vision unavailable and no fallback available.")
            
        if progress_callback:
            progress_callback("Running Gemini Vision...")
            
        try:
            gemini_engine = engines["gemini_vision"]
            res = gemini_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
            return res
        except RateLimitError as e:
            logger.warning(f"Gemini Vision rate limit: {e}")
            if progress_callback:
                reset_info = f" Resets in {e.reset_time}." if e.reset_time else ""
                progress_callback(f"⚠️ API limit reached.{reset_info} Falling back to PaddleOCR...")
            if "paddle" in engines:
                try:
                    paddle_engine = engines["paddle"]
                    res = paddle_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
                    if ai_correction:
                        res.metadata['needs_ai_correction'] = True
                    res.metadata['rate_limit'] = True
                    res.metadata['rate_limit_reset'] = e.reset_time
                    return res
                except Exception as e_pad:
                    logger.error(f"Fallback to PaddleOCR failed: {e_pad}")
                    return _fallback_result(f"API limit reached and PaddleOCR fallback failed.")
            return _fallback_result(f"API limit reached.{' Resets in ' + e.reset_time + '.' if e.reset_time else ''}")
        except Exception as e:
            logger.error(f"Gemini Vision failed: {e}")
            if progress_callback:
                progress_callback("Gemini Vision failed, falling back to PaddleOCR...")
            if "paddle" in engines:
                try:
                    paddle_engine = engines["paddle"]
                    res = paddle_engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
                    if ai_correction:
                        res.metadata['needs_ai_correction'] = True
                    return res
                except Exception as e_pad:
                    logger.error(f"Fallback to PaddleOCR failed: {e_pad}")
                    return _fallback_result(f"All engines failed.")
            return _fallback_result(f"Gemini Vision failed: {e}")
            
    else:
        logger.warning(f"Unknown OCR mode: {mode}")
        return _fallback_result(f"Unknown OCR mode: {mode}")

def _fallback_result(error_msg: str) -> OCRResult:
    return OCRResult(
        raw_text="",
        lines=[],
        confidence=0.0,
        engine="router",
        language="",
        metadata={'error': error_msg},
        layout_data=[]
    )

def benchmark_image(image_path: str, output_dir: str = 'benchmark',
                    progress_callback=None) -> Dict[str, Any]:
    """Run all available engines on an image and compare results."""
    engines = get_available_engines()
    out_path = Path(output_dir) / Path(image_path).stem
    out_path.mkdir(parents=True, exist_ok=True)
    
    lang = get_setting("ocr_lang", "ar")
    skip_tables = get_setting("skip_tables", False)
    
    results = {}
    ocr_results = {}
    
    for name, engine in engines.items():
        if progress_callback:
            progress_callback(f"Benchmarking engine: {name}...")
        try:
            res = engine.extract(image_path, languages=[lang], skip_tables=skip_tables)
            ocr_results[name] = res
            
            report = analyze_ocr_quality(res)
            
            # Save raw text
            text_file = out_path / f"{name}.txt"
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(res.raw_text)
                
            # Save metadata
            meta = {
                "engine": name,
                "confidence": res.confidence,
                "quality_score": report.score,
                "line_count": len(res.lines),
                "char_count": len(res.raw_text),
                "needs_fallback": report.needs_fallback
            }
            
            meta_file = out_path / f"{name}_metadata.json"
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4)
                
            results[name] = meta
            
        except Exception as e:
            logger.error(f"Engine {name} failed during benchmark: {e}")
            results[name] = {"error": str(e)}
            
    # Run validation between pairs if we have multiple
    if "paddle" in ocr_results and "gemini_vision" in ocr_results:
        if progress_callback:
            progress_callback("Running financial validation between engines...")
        val = validate_financial_numbers(
            ocr_results["gemini_vision"].raw_text,
            ocr_results["paddle"].raw_text
        )
        
        val_file = out_path / "financial_validation.json"
        with open(val_file, "w", encoding="utf-8") as f:
            json.dump(val, f, indent=4)
            
        results["financial_validation"] = {
            "match_rate": val.get("match_rate", 0.0),
            "matches": val.get("matches", 0),
            "discrepancies": len(val.get("discrepancies", []))
        }
        
    if progress_callback:
        progress_callback("Benchmark complete.")
        
    return results
