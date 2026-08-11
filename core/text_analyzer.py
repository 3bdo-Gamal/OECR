from typing import Dict

def calculate_counts_from_raw_text(raw_text: str, count_lines: bool = True, count_words: bool = False) -> Dict[str, int]:
    """
    Takes the raw string from PaddleOCR and calculates the total words and visual lines.
    """
    results = {"lines": 0, "words": 0, "characters": 0}
    if not raw_text or not raw_text.strip():
        return results
        
    if count_lines:
        lines = raw_text.strip().split('\n')
        valid_lines = [line for line in lines if line.strip()]
        results["lines"] = len(valid_lines)
    if count_words:
        words = raw_text.split()
        results["words"] = len(words)
        
    results["characters"] = len(raw_text.replace('\n', '').replace(' ', ''))
    return results