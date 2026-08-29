import logging
import re
from typing import List, Dict, Any

from core.ocr_base import OCRResult, QualityReport
from core.config import get_setting

logger = logging.getLogger(__name__)

def analyze_ocr_quality(result: OCRResult) -> QualityReport:
    """
    Analyze an OCR result and produce a quality score.
    """
    total_score = 0.0
    reasons = []

    # 1. Average confidence (40% of score)
    avg_conf = result.confidence
    if result.lines:
        avg_conf = sum(line.confidence for line in result.lines) / len(result.lines)
    
    total_score += avg_conf * 0.40
    if avg_conf < 0.7:
        reasons.append(f"Low average confidence: {avg_conf:.2f}")

    # 2. Low-confidence lines (15% of score)
    low_conf_score = 0.15
    if result.lines:
        low_conf_count = sum(1 for line in result.lines if line.confidence < 0.6)
        low_conf_ratio = low_conf_count / len(result.lines)
        if low_conf_ratio > 0.3:
            low_conf_score = 0.0
            reasons.append(f"High percentage of low-confidence lines: {low_conf_ratio:.1%}")
        elif low_conf_ratio > 0:
            low_conf_score = 0.15 * (1.0 - (low_conf_ratio / 0.3))
    total_score += low_conf_score

    # 3. Unicode issues (10% of score)
    if '\uFFFD' in result.raw_text:
        reasons.append("Contains Unicode replacement characters (\\uFFFD)")
    else:
        total_score += 0.10

    # 4. Suspicious sequences (10% of score)
    suspicious = False
    if re.search(r'[^\w\s]{3,}', result.raw_text):
        suspicious = True
        reasons.append("Contains suspicious sequences of punctuation (3+)")
    
    # Look for Latin characters mixed seamlessly inside Arabic words
    if re.search(r'(?:[a-zA-Z]+[\u0600-\u06FF]+)|(?:[\u0600-\u06FF]+[a-zA-Z]+)', result.raw_text):
        suspicious = True
        reasons.append("Contains random Latin characters mixed into Arabic words")

    if not suspicious:
        total_score += 0.10

    # 5. Token analysis (5% of score)
    tokens = result.raw_text.split()
    if any(len(t) > 50 for t in tokens):
        reasons.append("Contains extremely long tokens (>50 characters) suggesting merged words")
    else:
        total_score += 0.05

    # 6. Duplicate text detection (5% of score)
    lines = [line.strip() for line in result.raw_text.split('\n') if len(line.strip()) > 10]
    if lines and len(set(lines)) < len(lines) * 0.8:
        reasons.append("Contains repeated blocks of identical text")
    else:
        total_score += 0.05

    # 7. Empty/near-empty output (5% of score)
    if not result.raw_text or len(result.raw_text.strip()) < 20:
        total_score = 0.0
        reasons.append("Text is empty or extremely short (<20 characters)")
    else:
        total_score += 0.05

    # 8. Line fragmentation detection (10% of score)
    # PaddleOCR often splits Arabic text into hundreds of tiny fragments
    # A normal document page has 15-50 lines with 30+ chars each
    # 479 lines with 2790 chars (avg 5.8 chars/line) is extreme fragmentation
    all_lines = [l.strip() for l in result.raw_text.split('\n') if l.strip()]
    if all_lines:
        avg_chars_per_line = len(result.raw_text.replace('\n', '')) / len(all_lines)
        if avg_chars_per_line < 10:
            reasons.append(f"Extreme line fragmentation: avg {avg_chars_per_line:.1f} chars/line (expected 30+)")
        elif avg_chars_per_line < 20:
            total_score += 0.05
            reasons.append(f"Moderate line fragmentation: avg {avg_chars_per_line:.1f} chars/line")
        else:
            total_score += 0.10
    else:
        total_score += 0.10

    # 9. Arabic text coherence — CRITICAL signal
    # Detect garbled/reversed Arabic: if text has Arabic chars but very few spaces
    # relative to Arabic character count, words are merged/reversed (OCR artifact)
    # When this fires, it's a definitive sign of OCR failure — apply multiplicative penalty
    arabic_chars = sum(1 for c in result.raw_text if '\u0600' <= c <= '\u06FF' or '\uFB50' <= c <= '\uFDFF' or '\uFE70' <= c <= '\uFEFF')
    critical_arabic_issue = False
    if arabic_chars > 50:
        # For readable Arabic, expect roughly 1 space per 5-8 Arabic chars (ratio > 0.12)
        space_count = result.raw_text.count(' ')
        arabic_space_ratio = space_count / arabic_chars if arabic_chars > 0 else 0
        if arabic_space_ratio < 0.05:
            critical_arabic_issue = True
            reasons.append(f"CRITICAL: Arabic text is garbled/reversed (space ratio={arabic_space_ratio:.3f}, expected >0.12)")
        elif arabic_space_ratio < 0.10:
            total_score += 0.02
            reasons.append(f"Arabic text may have merged words (space ratio={arabic_space_ratio:.3f})")
        else:
            total_score += 0.05
    else:
        total_score += 0.05

    total_score = max(0.0, min(1.0, total_score))
    
    # Apply multiplicative penalty for critical issues
    # PaddleOCR reports high confidence on garbled Arabic text (confidently wrong)
    # so we need to override the confidence-based score
    if critical_arabic_issue:
        total_score *= 0.5  # Halve the score — garbled Arabic is unusable
    
    threshold = float(get_setting('quality_threshold', 0.65))
    needs_fallback = bool(total_score < threshold)
    
    numeric_tokens = extract_numeric_tokens(result.raw_text)

    return QualityReport(
        score=total_score,
        needs_fallback=needs_fallback,
        reasons=reasons,
        numeric_tokens=numeric_tokens
    )

def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract all numeric tokens from text preserving their exact original form.
    Matches standard digits, Arabic digits, dates, percentages, and decimals.
    """
    # \d matches both 0-9 and Arabic-Indic numerals in Python
    # [.,\-\/\u066B\u066C] covers typical separators including Arabic decimal/thousands separators
    pattern = r'(?:\d+(?:[.,\-\/\u066B\u066C]\d+)*%?)'
    return re.findall(pattern, text)

def validate_financial_numbers(primary_text: str, reference_text: str) -> dict:
    """
    Compare numeric tokens between two OCR results.
    Never modifies numbers, only flags positional discrepancies.
    """
    primary_numbers = extract_numeric_tokens(primary_text)
    reference_numbers = extract_numeric_tokens(reference_text)
    
    discrepancies = []
    matches = 0
    
    max_len = max(len(primary_numbers), len(reference_numbers))
    for i in range(max_len):
        prim = primary_numbers[i] if i < len(primary_numbers) else ""
        ref = reference_numbers[i] if i < len(reference_numbers) else ""
        
        if prim == ref and prim != "":
            matches += 1
        else:
            discrepancies.append({
                'position': i,
                'primary': prim,
                'reference': ref
            })
            
    match_rate = float(matches / max_len) if max_len > 0 else 1.0
    
    return {
        'primary_numbers': primary_numbers,
        'reference_numbers': reference_numbers,
        'matches': matches,
        'discrepancies': discrepancies,
        'match_rate': match_rate
    }
