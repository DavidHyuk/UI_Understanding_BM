import re
import json
import difflib
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM
from datasets import load_dataset

def parse_coords(text):
    """Parses coordinates from text (e.g. <loc0500>, 0.5, or JSON box_2d)"""
    # Try parsing JSON first
    try:
        # Extract JSON part if mixed with text
        json_match = re.search(r'\[.*\]', text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            if isinstance(data, list) and len(data) > 0:
                # Look for box_2d in the first item
                if "box_2d" in data[0]:
                    coords = data[0]["box_2d"]
                    # Check if normalized or integer
                    if any(c > 1.0 for c in coords):
                        # Match token-based scale (usually 1024 for Gemma/PaliGemma)
                        return [c / 1024.0 for c in coords]
                    return coords
    except:
        pass

    loc_tokens = re.findall(r"<loc(\d{3,4})>", text)
    if loc_tokens:
        return [int(t) / 1024.0 for t in loc_tokens]
    floats = re.findall(r"0\.\d+", text)
    if floats:
        return [float(f) for f in floats]
    return []

def eval_screenspot(pred_text, gt_data):
    if not gt_data: return {"success": 0.0}
    gt_bbox = gt_data if isinstance(gt_data, list) and len(gt_data) == 4 else None
    if not gt_bbox: return {"success": 0.0}
    pred_coords = parse_coords(pred_text)
    pred_point = None
    if len(pred_coords) >= 2:
        y, x = pred_coords[0], pred_coords[1]
        pred_point = [x, y]
    elif len(pred_coords) >= 4:
        y1, x1, y2, x2 = pred_coords[:4]
        pred_point = [(x1 + x2) / 2, (y1 + y2) / 2]
    if not pred_point: return {"success": 0.0}
    px, py = pred_point
    xmin, ymin, xmax, ymax = gt_bbox
    if xmin <= px <= xmax and ymin <= py <= ymax:
        return {"success": 1.0}
    return {"success": 0.0}

def eval_sroie(pred_text, gt_text):
    def normalize(s):
        if not s: return ""
        s = re.sub(r"<[^>]+>", " ", s)
        return re.sub(r"\s+", " ", s.lower().strip())
    p = normalize(pred_text)
    g = normalize(gt_text)
    
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

def parse_sroie_tags(text):
    """Parses SROIE tags into a dictionary."""
    if not text:
        return {}
    # Find all <(tag)>value</(tag)>
    matches = re.findall(r"<(s_[^>]+)>([^<]*)</\1>", text)
    return {tag: val.strip() for tag, val in matches}

def postprocess_sroie_raw(text):
    """Heuristically parses raw SROIE output (newline or space separated) into a dict."""
    if not text:
        return {}
    
    # Try splitting by newlines first (preferred from new prompt)
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

    # Fallback to space splitting if newlines missing (heuristic)
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

# Configuration for datasets
DATASET_CONFIGS = {
    "screenspot": {
        "id": "HongxinLi/ScreenSpot_v2",
        "split": "test",
        "prompt_fn": lambda ex: f"Detect the specific element described: {ex['instruction']} Output the bounding box coordinates for this element only.",
        "gt_fn": lambda ex: ex.get("bbox") or ex.get("point", "N/A"),
        "eval_fn": eval_screenspot
    },
    "sroie": {
        "id": "rajistics/sroie",
        "split": "train",
        "prompt_fn": lambda ex: "Extract the total amount, date, company name, and address. Output only the values on separate lines. Do not include labels or field names.",
        "gt_fn": lambda ex: ex.get("text", "N/A"),
        "eval_fn": eval_sroie
    }
}

def load_model_and_processor(model_id, device):
    """Loads the model and processor."""
    print(f"Loading model: {model_id}...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device, # Use direct device or auto if mapped
        trust_remote_code=True
    )
    return model, processor

def get_dataset(dataset_name):
    """Loads the dataset based on configuration."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    config = DATASET_CONFIGS[dataset_name]
    try:
        dataset = load_dataset(config["id"], split=config["split"], trust_remote_code=True)
    except Exception as e:
        print(f"Warning: {e}. Retrying without trust_remote_code...")
        dataset = load_dataset(config["id"], split=config["split"])
    return dataset

