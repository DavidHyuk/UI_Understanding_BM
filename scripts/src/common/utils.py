import re
import json
import difflib
import sys
import os
from collections import Counter
from datasets import load_dataset
from PIL import Image, ImageDraw

# Import ModelFactory
try:
    # Try relative import first (when used as package)
    from .model_wrappers import ModelFactory
except ImportError:
    try:
        # Try absolute import (when running from root)
        from scripts.src.common.model_wrappers import ModelFactory
    except ImportError:
        # Fallback for when running standalone or path issues
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model_wrappers import ModelFactory

try:
    import evaluate
    import nltk
    # Ensure nltk resources are available
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    METRICS_LOADED = True
except ImportError:
    METRICS_LOADED = False

def parse_coords(text, unscaled=False):
    """Parses coordinates from text (e.g. <loc0500>, 0.5, or JSON box_2d)"""
    # Try parsing JSON first
    try:
        # Extract JSON part if mixed with text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, list) and len(data) > 0:
                # Case 1: List of numbers (or stringified numbers) e.g. [10, 20, 30, 40] or ["10", "20", "30", "40"]
                # Filter out any non-numeric items to be safe, but try to convert strings
                try:
                    # Check if it's a flat list of potential coordinates
                    flat_coords = []
                    is_flat_coords = True
                    for x in data:
                        if isinstance(x, (int, float)):
                            flat_coords.append(float(x))
                        elif isinstance(x, str):
                            # Handle "10, 20, 30, 40" as a single item
                            sub_parts = x.replace(',', ' ').split()
                            for sp in sub_parts:
                                try:
                                    flat_coords.append(float(sp))
                                except ValueError:
                                    is_flat_coords = False
                                    break
                        else:
                             is_flat_coords = False
                        
                        if not is_flat_coords: break
                    
                    if is_flat_coords and len(flat_coords) >= 2:
                        # Valid coordinate list found directly in JSON
                        if not unscaled and any(c > 1.0 for c in flat_coords):
                            return [c / 1024.0 for c in flat_coords]
                        return flat_coords
                except:
                    pass

                # Case 2: Structured output (list of dicts)
                item = data[0]
                if isinstance(item, dict):
                    coords = item.get("box_2d") or item.get("bounding_box")
                    
                    if coords:
                        # Handle list of dicts (e.g. [{"ymin":...}]) -> take first
                        if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], dict):
                            coords = coords[0]

                        # Handle dictionary output (convert to list [ymin, xmin, ymax, xmax])
                        if isinstance(coords, dict):
                            y1 = coords.get("ymin") if "ymin" in coords else coords.get("y1")
                            x1 = coords.get("xmin") if "xmin" in coords else coords.get("x1")
                            y2 = coords.get("ymax") if "ymax" in coords else coords.get("y2")
                            x2 = coords.get("xmax") if "xmax" in coords else coords.get("x2")
                            
                            if y1 is not None and x1 is not None and y2 is not None and x2 is not None:
                                coords = [y1, x1, y2, x2]
                            else:
                                # Fallback: assume values are coordinates if keys don't match known patterns
                                coords = list(coords.values())

                        # Check if normalized or integer
                        # Robustly filter to ensure only numbers
                        coords = [c for c in coords if isinstance(c, (int, float))]

                        if not unscaled and any(c > 1.0 for c in coords):
                            # Match token-based scale (usually 1024 for Gemma/PaliGemma)
                            return [c / 1024.0 for c in coords]
                        return coords
                    
                    # Check for point coordinate
                    if "point" in item or "coordinate" in item:
                        coords = item.get("point") or item.get("coordinate")
                        if coords:
                            # Handle list of dicts
                            if isinstance(coords, list) and len(coords) > 0 and isinstance(coords[0], dict):
                                coords = coords[0]

                            # Handle dictionary output (convert to list [y, x])
                            if isinstance(coords, dict):
                                y = coords.get("y") if "y" in coords else coords.get("row")
                                x = coords.get("x") if "x" in coords else coords.get("col")
                                if y is not None and x is not None:
                                    coords = [y, x]
                                else:
                                    coords = list(coords.values())

                            # Robustly filter to ensure only numbers
                            coords = [c for c in coords if isinstance(c, (int, float))]

                            if not unscaled and any(c > 1.0 for c in coords):
                                return [c / 1024.0 for c in coords]
                            return coords
    except:
        pass

    loc_tokens = re.findall(r"<loc(\d{3,4})>", text)
    if loc_tokens:
        if not unscaled:
            return [int(t) / 1024.0 for t in loc_tokens]
        else:
            return [float(t) for t in loc_tokens]

    # Handle both 0.123 and integer/raw numbers in brackets/text
    floats = re.findall(r"(?:0\.\d+|\d+)", text)
    if floats:
        # If there are many numbers, filter for those likely to be coords (0-1024)
        results = []
        for f in floats:
            val = float(f)
            if not unscaled and val > 1.0:
                results.append(val / 1024.0)
            else:
                results.append(val)
        return results
    return []

def eval_screenspot(pred_text, gt_data, image_size=None, **kwargs):
    if not gt_data: return {"success": 0.0, "iou": 0.0}
    gt_bbox = gt_data if isinstance(gt_data, list) and len(gt_data) == 4 else None
    if not gt_bbox: return {"success": 0.0, "iou": 0.0}
    
    # Use unscaled=True to handle normalization manually based on image size
    pred_coords = parse_coords(pred_text, unscaled=True)
    
    # Normalize if needed
    if any(c > 1.0 for c in pred_coords):
        if image_size:
            w, h = image_size
            # Assume [y1, x1, y2, x2] (YXYX) or [y, x] (YX)
            norm_coords = []
            for i, c in enumerate(pred_coords):
                if i % 2 == 0: # y
                    norm_coords.append(c / h if h > 0 else 0)
                else: # x
                    norm_coords.append(c / w if w > 0 else 0)
            pred_coords = norm_coords
        else:
            # Fallback to 1024 scale if image size not provided
            pred_coords = [c / 1024.0 for c in pred_coords]

    pred_point = None
    iou = 0.0
    
    if len(pred_coords) >= 4:
        # Box output case: [y1, x1, y2, x2]
        y1, x1, y2, x2 = pred_coords[:4]
        pred_point = [(x1 + x2) / 2, (y1 + y2) / 2]
        
        # Calculate IoU
        # Pred: y1, x1, y2, x2 -> x1, y1, x2, y2
        p_xmin, p_ymin, p_xmax, p_ymax = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        # GT: xmin, ymin, xmax, ymax
        g_xmin, g_ymin, g_xmax, g_ymax = gt_bbox
        
        # Intersection
        xi1 = max(p_xmin, g_xmin)
        yi1 = max(p_ymin, g_ymin)
        xi2 = min(p_xmax, g_xmax)
        yi2 = min(p_ymax, g_ymax)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        # Union
        p_area = (p_xmax - p_xmin) * (p_ymax - p_ymin)
        g_area = (g_xmax - g_xmin) * (g_ymax - g_ymin)
        union_area = p_area + g_area - inter_area
        
        if union_area > 0:
            iou = inter_area / union_area
            
    elif len(pred_coords) >= 2:
        # Point output case: [y, x]
        y, x = pred_coords[0], pred_coords[1]
        pred_point = [x, y]
        # IoU is typically 0 for a point, or could be treated as a small box match if we wanted
        # but standard IoU is 0.
    
    if not pred_point: return {"success": 0.0, "iou": 0.0}
    
    px, py = pred_point
    xmin, ymin, xmax, ymax = gt_bbox
    
    success = 1.0 if (xmin <= px <= xmax and ymin <= py <= ymax) else 0.0
    
    return {"success": success, "iou": iou}

def eval_sroie(pred_text, gt_text, **kwargs):
    # Try to extract structured content to compare values only
    p_data = parse_sroie_tags(pred_text)
    if not p_data:
        p_data = postprocess_sroie_raw(pred_text)
        
    g_data = parse_sroie_tags(gt_text)
    if not g_data:
        g_data = postprocess_sroie_raw(gt_text)
        
    # If we have structured data for both, construct strings from values
    if p_data and g_data:
        keys = ["s_total", "s_date", "s_company", "s_address"]
        p_list = [str(p_data.get(k, "")).strip() for k in keys]
        g_list = [str(g_data.get(k, "")).strip() for k in keys]
        p_raw = " ".join(p_list)
        g_raw = " ".join(g_list)
    else:
        p_raw = pred_text
        g_raw = gt_text

    def normalize(s):
        if not s: return ""
        # Remove XML tags
        s = re.sub(r"<[^>]+>", " ", s)
        # Remove key labels like "s_total:", "Total:", etc.
        s = re.sub(r"\b(?:s_)?(?:total|date|company|address):", " ", s, flags=re.IGNORECASE)
        return re.sub(r"\s+", " ", s.lower().strip())
        
    p = normalize(p_raw)
    g = normalize(g_raw)
    
    p_words = p.split()
    g_words = g.split()
    
    # WER Calculation (Levenshtein on words)
    def word_edit_distance(ref, hyp):
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1): dp[i][0] = i
        for j in range(n + 1): dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        return dp[m][n]

    dist = word_edit_distance(g_words, p_words)
    wer = dist / len(g_words) if g_words else (0.0 if not p_words else 1.0)

    # F1 Calculation
    if not g_words: return {"wer": wer, "f1": 0.0}
    common = 0
    g_words_copy = list(g_words)
    for word in p_words:
        if word in g_words_copy:
            common += 1
            g_words_copy.remove(word)
    precision = common / len(p_words) if p_words else 0.0
    recall = common / len(g_words) if g_words else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"wer": wer, "f1": f1}

def levenshtein_distance(s1, s2):
    """Character-level Levenshtein distance."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def eval_screenqa(pred_text, gt_text, **kwargs):
    """Evaluates ScreenQA using ANLS and Normalized F1."""
    def normalize(s):
        if not s: return ""
        s = s.lower().strip()
        # Replace punctuation with space to avoid merging words, then collapse spaces
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    p = normalize(pred_text)
    # Remove comma and periods specifically for substring matching if they are at the end
    p_clean = re.sub(r'[,.\s]+$', '', p).strip()
    # Create word set for and-based matching (for multiple items)
    p_words = set(p.split())
    
    if isinstance(gt_text, list):
        if not gt_text: return {"anls": 0.0, "f1": 0.0}
        results = [eval_screenqa(p, gt) for gt in gt_text]
        return {
            "anls": max(r["anls"] for r in results),
            "f1": max(r["f1"] for r in results)
        }
        
    g = normalize(gt_text)
    
    # ANLS Calculation
    if not g:
        anls = 0.0
    else:
        g_clean = re.sub(r'[,.\s]+$', '', g).strip()
        g_words = set(g.split())
        # Check if one is a substring of the other (common in VQA/ScreenQA)
        # OR if all words in p are in g (for multiple items)
        if p_clean in g or g_clean in p or p in g or g in p or (p_words and p_words.issubset(g_words)):
            anls = 1.0
        else:
            dist = levenshtein_distance(p, g)
            anls = 1.0 - (dist / max(len(p), len(g)))
            if anls < 0.5: anls = 0.0
        
    # F1 Calculation
    p_words = p.split()
    g_words = g.split()
    if not p_words or not g_words:
        f1 = 1.0 if not p_words and not g_words else 0.0
    else:
        common = Counter(p_words) & Counter(g_words)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(p_words)
            recall = num_same / len(g_words)
            f1 = 2 * (precision * recall) / (precision + recall)
    
    return {"anls": anls, "f1": f1}

def eval_captioning(pred_text, gt_text, **kwargs):
    """Evaluates Captioning using ROUGE-L and placeholders for CIDEr/METEOR."""
    def normalize(s):
        if not s: return ""
        s = s.lower().strip()
        # Remove bold markers and special symbols
        s = re.sub(r'\*+', '', s)
        s = re.sub(r'[^\w\s]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s
    
    p = normalize(pred_text)
    
    if isinstance(gt_text, list):
        if not gt_text: return {"rouge_l": 0.0}
        results = [eval_captioning(p, gt) for gt in gt_text]
        return {
            "rouge_l": max(r["rouge_l"] for r in results),
            "functional_match": max(r.get("functional_match", 0.0) for r in results)
        }
        
    g = normalize(gt_text)
    
    functional_match = 0.0
    # Check for substring match (High correlation for widget functional descriptions)
    if p and g and (p in g or g in p):
        functional_match = 1.0
    
    # Check for word overlap (Keyword based) for functional match
    p_words = set(p.split())
    g_words = set(g.split())
    if p_words and g_words and functional_match == 0.0:
        common = p_words.intersection(g_words)
        # If overlap is significant (>= 50% of the shorter phrase), consider it a functional match
        # e.g. "search" vs "search locations" -> 1/1 = 1.0
        if len(common) / min(len(p_words), len(g_words)) >= 0.5:
            functional_match = 1.0
            
    # Try stemming if simple overlap fails (e.g. "Closes" vs "Close")
    if functional_match == 0.0 and METRICS_LOADED:
        try:
            from nltk.stem import PorterStemmer
            stemmer = PorterStemmer()
            p_stems = set(stemmer.stem(w) for w in p_words)
            g_stems = set(stemmer.stem(w) for w in g_words)
            if p_stems and g_stems:
                common_stems = p_stems.intersection(g_stems)
                if len(common_stems) / min(len(p_stems), len(g_stems)) >= 0.5:
                    functional_match = 1.0
        except Exception:
            pass
            
    # Check using UI Synonyms if still no match
    if functional_match == 0.0:
        def get_synonyms(word):
            # Check direct mapping
            syns = set()
            if word in UI_SYNONYMS:
                syns.update(UI_SYNONYMS[word])
            # Check if word is a value in any key (reverse mapping)
            for k, v in UI_SYNONYMS.items():
                if word in v:
                    syns.add(k)
            return syns

        # Count how many p_words match g_words (either directly or via synonym)
        match_count = 0
        p_words_list = list(p_words)
        
        for pw in p_words_list:
            matched = False
            # Direct or Stem match (already checked? No, strict intersection was checked)
            if pw in g_words:
                matched = True
            else:
                # Synonym match
                pw_syns = get_synonyms(pw)
                if not pw_syns.isdisjoint(g_words):
                    matched = True
            
            if matched:
                match_count += 1
                
        # Also check reverse (g words covered by p)
        g_match_count = 0
        g_words_list = list(g_words)
        for gw in g_words_list:
            matched = False
            if gw in p_words:
                matched = True
            else:
                gw_syns = get_synonyms(gw)
                if not gw_syns.isdisjoint(p_words):
                    matched = True
            if matched:
                g_match_count += 1
                
        # Calculate overlap ratio based on max coverage
        p_len = len(p_words)
        g_len = len(g_words)
        
        if p_len > 0 and g_len > 0:
            # We use max of (p_coverage, g_coverage) against min_len logic
            # But adhering to previous logic: matches / min_len >= 0.5
            
            # Use the better match count
            best_match = max(match_count, g_match_count)
            
            if best_match / min(p_len, g_len) >= 0.5:
                functional_match = 1.0

    # ROUGE-L (Longest Common Subsequence)
    def lcs(X, Y):
        m, n = len(X), len(Y)
        L = [[0] * (n + 1) for i in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0 or j == 0: L[i][j] = 0
                elif X[i-1] == Y[j-1]: L[i][j] = L[i-1][j-1] + 1
                else: L[i][j] = max(L[i-1][j], L[i][j-1])
        return L[m][n]

    p_tokens = p.split()
    g_tokens = g.split()
    
    if not p_tokens or not g_tokens:
        rouge_l = 0.0
    else:
        lcs_val = lcs(p_tokens, g_tokens)
        precision = lcs_val / len(p_tokens)
        recall = lcs_val / len(g_tokens)
        rouge_l = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        
    return {
        "rouge_l": rouge_l,
        "functional_match": functional_match
    }

def calculate_corpus_metrics(results, dataset_name):
    """Calculates corpus-level metrics like CIDEr and METEOR."""
    if not METRICS_LOADED:
        return {}
    
    if dataset_name != "widget_captioning":
        return {}

    preds = [r["prediction"] for r in results]
    # Handle ground truth which might be a list or a string
    refs = [r["ground_truth"] if isinstance(r["ground_truth"], list) else [r["ground_truth"]] for r in results]
    
    corpus_results = {}
    
    try:
        # 1. METEOR
        meteor = evaluate.load("meteor")
        res = meteor.compute(predictions=preds, references=refs)
        corpus_results["meteor"] = res["meteor"]
        
        # 2. ROUGE-L (via evaluate)
        rouge = evaluate.load("rouge")
        res = rouge.compute(predictions=preds, references=refs)
        corpus_results["rougeL_corpus"] = res["rougeL"]
        
        # 3. CIDEr (Usually requires pycocoevalcap)
        try:
            # Note: CIDEr is often part of community metrics or specific implementations
            cider = evaluate.load("cider")
            res = cider.compute(predictions=preds, references=refs)
            corpus_results["cider"] = res["avg_cider"] if "avg_cider" in res else res.get("cider")
        except:
            # If evaluate's cider fails, we try to use pycocoevalcap directly if available
            try:
                from pycocoevalcap.cider.cider import Cider
                scorer = Cider()
                # CIDEr expects dict {id: [list of refs]} and {id: [prediction]}
                dict_preds = {i: [p] for i, p in enumerate(preds)}
                dict_refs = {i: r for i, r in enumerate(refs)}
                score, _ = scorer.compute_score(dict_refs, dict_preds)
                corpus_results["cider"] = score
            except ImportError:
                corpus_results["cider_status"] = "CIDEr requires pycocoevalcap"
            
    except Exception as e:
        print(f"Warning: Error calculating corpus metrics: {e}")
        
    return corpus_results

def parse_sroie_tags(text):
    """Parses SROIE tags into a dictionary."""
    if not text:
        return {}
    # Find all <(tag)>value</(tag)>
    matches = re.findall(r"<(s_[^>]+)>([^<]*)</\1>", text)
    return {tag: val.strip() for tag, val in matches}

def postprocess_sroie_raw(text):
    """Heuristically parses raw SROIE output into a dict."""
    if not text:
        return {}

    # 1. Try parsing Key: Value pairs (Robust)
    data = {}
    lines = text.strip().split('\n')
    current_key = None
    
    # Regex for strict keys at start of line
    key_pattern = re.compile(r"^(Total|Date|Company|Address):?\s*(.*)", re.IGNORECASE)
    
    keys_found = 0
    for line in lines:
        line = line.strip()
        if not line: continue
        
        match = key_pattern.match(line)
        if match:
            key_name = match.group(1).lower()
            value = match.group(2).strip()
            
            if key_name == "total":
                data["s_total"] = value
                current_key = "s_total"
            elif key_name == "date":
                data["s_date"] = value
                current_key = "s_date"
            elif key_name == "company":
                data["s_company"] = value
                current_key = "s_company"
            elif key_name == "address":
                data["s_address"] = value
                current_key = "s_address"
            keys_found += 1
        elif current_key == "s_address":
            # Append to address if it looks like continuation (common for address)
            if "s_address" in data:
                data["s_address"] += " " + line
            else:
                data["s_address"] = line

    if keys_found >= 2:
        # Fill missing keys with empty string
        for k in ["s_total", "s_date", "s_company", "s_address"]:
            if k not in data:
                data[k] = ""
        return data

    # 2. Fallback: Try splitting by newlines (Old behavior / Line-based)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    
    if len(lines) >= 3:
        # High confidence that model followed newline instruction
        s_total = lines[0]
        s_date = lines[1]
        
        remaining = lines[2:]
        if len(remaining) == 1:
             s_company = remaining[0]
             s_address = ""
        else:
             s_company = remaining[0]
             s_address = " ".join(remaining[1:]) # Join rest as address
             
        return {
            "s_total": s_total,
            "s_date": s_date,
            "s_company": s_company,
            "s_address": s_address
        }

    # 3. Fallback to space splitting if newlines missing (heuristic)
    parts = text.split()
    if len(parts) < 2:
        return {"raw_output": text}
        
    total = parts[0]
    date = parts[1]
    remaining = parts[2:]
    
    # Heuristic to split Company and Address
    keywords = ["NO", "NO.", "LOT", "JALAN", "BATU", "BLOCK", "BLK", "LEVEL", "SUITE", "UNIT"]
    split_index = -1
    
    for i, token in enumerate(remaining):
        clean_t = token.upper().replace(",", "").replace(".", "")
        if clean_t in keywords:
            split_index = i
            break
        if re.match(r"^\d+,$", token):
             pass
             
    if split_index != -1:
        company = " ".join(remaining[:split_index])
        address = " ".join(remaining[split_index:])
    else:
        if remaining:
             company = " ".join(remaining)
             address = "(Parsing failed)"
        else:
             company = ""
             address = ""
             
    return {
        "s_total": total,
        "s_date": date,
        "s_company": company,
        "s_address": address
    }

def format_sroie_display(text):
    """Formats SROIE tags into a readable 'key: value' string."""
    parsed = parse_sroie_tags(text)
    if parsed:
        return "\n".join([f"{tag}: {val}" for tag, val in parsed.items()])
    return text

def highlight_text_diff(pred, gt):
    """
    Compares pred against gt and returns HTML with differences in pred highlighted in red.
    """
    if not pred: return ""
    if not gt: return f"<span style='color:#ff4b4b'>{pred}</span>"
    
    # Use SequenceMatcher to find differences at character level
    matcher = difflib.SequenceMatcher(None, pred, gt)
    html = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        segment = pred[i1:i2]
        if tag == 'equal':
            html.append(segment)
        elif tag == 'replace':
            # Pred has different content -> Red
            html.append(f"<span style='color:#ff4b4b'>{segment}</span>")
        elif tag == 'delete':
            # Pred has extra content -> Red
            html.append(f"<span style='color:#ff4b4b'>{segment}</span>")
        elif tag == 'insert':
            # Pred is missing content -> Nothing to show in Pred string
            pass
            
    return "".join(html)

def highlight_gt_diff(gt, pred):
    """
    Compares gt against pred and returns HTML with missing parts in gt highlighted.
    """
    if not gt: return ""
    if not pred: return f"<span style='color:#ff4b4b'>{gt}</span>"
    
    matcher = difflib.SequenceMatcher(None, gt, pred)
    html = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        segment = gt[i1:i2]
        if tag == 'equal':
            html.append(segment)
        elif tag == 'replace':
            # GT has content that was replaced in Pred (mismatch) -> Red
            html.append(f"<span style='color:#ff4b4b'>{segment}</span>")
        elif tag == 'delete':
            # GT has content that is missing in Pred -> Red
            html.append(f"<span style='color:#ff4b4b'>{segment}</span>")
        elif tag == 'insert':
            # Pred has extra content -> Not in GT string
            pass
            
    return "".join(html)

# Custom UI Synonyms for semantic evaluation
UI_SYNONYMS = {
    "dismiss": {"close", "cancel", "exit", "hide", "remove", "back"},
    "close": {"dismiss", "cancel", "exit", "shut", "hide", "stop"},
    "cancel": {"dismiss", "close", "abort", "stop", "back"},
    "delete": {"remove", "trash", "discard", "erase", "clear"},
    "remove": {"delete", "trash", "discard", "erase", "clear"},
    "edit": {"change", "modify", "update", "pen", "write"},
    "add": {"create", "new", "plus", "insert", "append"},
    "create": {"add", "new", "make", "generate"},
    "search": {"find", "query", "look", "explore"},
    "find": {"search", "query", "look", "locate"},
    "settings": {"preferences", "options", "config", "configuration", "gear", "setup"},
    "previous": {"back", "last", "undo"},
    "back": {"previous", "return", "cancel", "dismiss"},
    "next": {"forward", "continue", "proceed", "go"},
    "favorite": {"like", "heart", "save", "bookmark", "star"},
    "share": {"send", "export", "forward"},
    "menu": {"options", "hamburger", "drawer", "list"},
    "home": {"start", "main", "dashboard"},
    "location": {"place", "address", "map", "area"},
    "place": {"location", "address", "spot"},
    "image": {"photo", "picture", "pic"},
    "photo": {"image", "picture", "pic"},
    "screen": {"page", "window", "view"},
    "page": {"screen", "window", "view"},
    "window": {"screen", "page", "view"}
}

def draw_bbox_on_image(example):
    """Draws a bounding box on the image if 'bbox' is present."""
    image = example["image"]
    bbox = example.get("bbox")
    if not bbox or not image:
        return image
        
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # Assume [xmin, ymin, xmax, ymax] normalized 0-1
    xmin, ymin, xmax, ymax = bbox
    coords = [xmin * w, ymin * h, xmax * w, ymax * h]
    
    draw.rectangle(coords, outline="red", width=5)
    return img

# Configuration for datasets
DATASET_CONFIGS = {
    "screenspot": {
        "id": "HongxinLi/ScreenSpot_v2",
        "local_path": "data/screenspot_v2",
        "split": "test",
        "prompt_fn": lambda ex: f"Detect the specific element described: {ex['instruction']} Output the bounding box coordinates for this element only. Output valid JSON in the format [ymin, xmin, ymax, xmax].",
        "gt_fn": lambda ex: ex.get("bbox") or ex.get("point", "N/A"),
        "instruction_key": "instruction",
        "eval_fn": eval_screenspot
    },
    "sroie": {
        "id": "rajistics/sroie",
        "local_path": "data/sroie",
        "split": "train",
        "prompt_fn": lambda ex: "Extract the total amount, date, company name, and address. Output in the following format:\nTotal: {amount}\nDate: {date}\nCompany: {name}\nAddress: {address}",
        "gt_fn": lambda ex: ex.get("text", "N/A"),
        "instruction_key": None,
        "eval_fn": eval_sroie
    },
    "screenqa": {
        "id": "rootsautomation/RICO-ScreenQA",
        "local_path": "data/screenqa",
        "split": "test",
        "prompt_fn": lambda ex: f"Answer the question based on the screen. Question: {ex['question']} Answer as briefly as possible.",
        "gt_fn": lambda ex: [gt["full_answer"] for gt in ex["ground_truth"]] if "ground_truth" in ex and isinstance(ex["ground_truth"], list) and len(ex["ground_truth"]) > 0 and "full_answer" in ex["ground_truth"][0] else (ex.get("answer") or ex.get("answers") or "N/A"),
        "instruction_key": "question",
        "eval_fn": eval_screenqa
    },
    "widget_captioning": {
        "id": "rootsautomation/RICO-WidgetCaptioning",
        "local_path": "data/widget_captioning",
        "split": "test",
        "prompt_fn": lambda ex: "Describe the function of the highlighted UI element in a short phrase.",
        "gt_fn": lambda ex: ex.get("captions") or ex.get("caption") or ex.get("text", "N/A"),
        "instruction_key": None,
        "eval_fn": eval_captioning,
        "transform_fn": draw_bbox_on_image
    }
}

def load_model_and_processor(model_id, device):
    """Loads the model and processor using the Unified Strategy Pattern."""
    print(f"Loading model: {model_id} via ModelFactory...")
    
    wrapper = ModelFactory.get_model(model_id, device)
    
    # Return wrapper as 'model' and its processor
    # This allows clients to use model.generate_content(...) which is the new unified API
    # while still having access to processor if absolutely needed (though they shouldn't need it for generation anymore)
    return wrapper, wrapper.get_processor()

from datasets import load_from_disk

def get_dataset(dataset_name):
    """Loads the dataset based on configuration, preferring local path if available."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    local_path = config.get("local_path")
    
    if local_path and os.path.exists(local_path):
        print(f"Loading dataset {dataset_name} from local path: {local_path}")
        try:
            dataset = load_from_disk(local_path)
            # If sharded/split in save_to_disk, we might need to select the split
            # but usually save_to_disk saves the whole object.
            # If it's a DatasetDict, select the split.
            if hasattr(dataset, "keys") and config["split"] in dataset:
                return dataset[config["split"]]
            return dataset
        except Exception as e:
            print(f"Failed to load from local path {local_path}: {e}. Falling back to HF Hub.")

    print(f"Loading dataset {dataset_name} from HF Hub: {config['id']}")
    try:
        dataset = load_dataset(config["id"], split=config["split"], trust_remote_code=True)
    except Exception as e:
        print(f"Warning: {e}. Retrying without trust_remote_code...")
        dataset = load_dataset(config["id"], split=config["split"])
    return dataset
