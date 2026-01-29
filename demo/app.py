import streamlit as st
import torch
import time
import os
from PIL import Image, ImageDraw
import numpy as np

# Import from utils
from utils import load_model_and_processor, get_dataset, DATASET_CONFIGS, parse_coords, format_sroie_display, parse_sroie_tags, postprocess_sroie_raw, highlight_text_diff, highlight_gt_diff

# Page Configuration
st.set_page_config(
    page_title="Multimodal AI Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-size: 2.5rem;
        color: #4B4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #666;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.title("Configuration")

model_id = st.sidebar.text_input("Model ID", "models/gemma-3n")
device_option = st.sidebar.selectbox("Device", ["cuda", "cpu"], index=0 if torch.cuda.is_available() else 1)
dataset_name = st.sidebar.selectbox("Dataset", list(DATASET_CONFIGS.keys()))

st.sidebar.markdown("---")
st.sidebar.info(f"Detected Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# --- Resource Loading (Cached) ---
@st.cache_resource
def load_resources(model_path, device_name):
    return load_model_and_processor(model_path, device_name)

@st.cache_resource
def load_data(ds_name):
    return get_dataset(ds_name)

# --- Main App ---
st.markdown('<div class="main-header">🧠 Multimodal AI Interactive Eval</div>', unsafe_allow_html=True)

try:
    with st.spinner(f"Loading Model '{model_id}'..."):
        model, processor = load_resources(model_id, device_option)
    
    with st.spinner(f"Loading Dataset '{dataset_name}'..."):
        dataset = load_data(dataset_name)
        
    st.success(f"Loaded Model & Dataset ({len(dataset)} samples)")

except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.stop()

# --- Sample Selection ---
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    sample_idx = st.number_input("Sample Index", min_value=0, max_value=len(dataset)-1, value=0, step=1)
    
    if st.button("Random Sample"):
        sample_idx = np.random.randint(0, len(dataset))
        st.rerun() # Use rerun to update the number input if possible, or just update state

# Get Sample
example = dataset[sample_idx]
image = example["image"]
dataset_config = DATASET_CONFIGS[dataset_name]
prompt_text = dataset_config["prompt_fn"](example)
ground_truth = dataset_config["gt_fn"](example)

# --- Layout: Input vs Output ---
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Input")
    st.image(image, use_container_width=True, caption=f"Sample {sample_idx}")
    
    with st.expander("Show Prompt", expanded=True):
        st.write(prompt_text)

with col_right:
    st.subheader("Inference & Results")
    
    if st.button("Run Inference", type="primary"):
        start_time = time.time()
        
        # Prepare Chat
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt_text}
                ]
            }
        ]
        
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        # Inference
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
        inputs = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=512,
                do_sample=False
            )
            
        elapsed = time.time() - start_time
        input_len = inputs["input_ids"].shape[1]
        should_skip_special = (dataset_name == "sroie")
        generated_text = processor.decode(generated_ids[0][input_len:], skip_special_tokens=should_skip_special)
        
        # Evaluation
        metrics = dataset_config["eval_fn"](generated_text, ground_truth)
        
        # Display Results
        st.markdown(f"**Inference Time:** {elapsed:.2f}s")
        
        st.markdown("### Generated Output")
        if dataset_name == "sroie":
            pred_tags = parse_sroie_tags(generated_text)
            if not pred_tags:
                pred_tags = postprocess_sroie_raw(generated_text)
            
            gt_tags = parse_sroie_tags(ground_truth)
            
            if pred_tags:
                for tag, val in pred_tags.items():
                    gt_val = gt_tags.get(tag, "")
                    if val != gt_val:
                        highlighted_val = highlight_text_diff(val, gt_val)
                        st.markdown(f"**{tag}**: {highlighted_val}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{tag}**: {val}", unsafe_allow_html=True)
            else:
                st.write(generated_text)
        else:
            st.code(generated_text, language="text")
            
        if dataset_name == "sroie":
            st.markdown("### Ground Truth")
            gt_tags = parse_sroie_tags(ground_truth)
            if gt_tags:
                for tag, gt_val in gt_tags.items():
                    pred_val = pred_tags.get(tag, "") if pred_tags else ""
                    if pred_val != gt_val:
                        highlighted_gt = highlight_gt_diff(gt_val, pred_val)
                        st.markdown(f"**{tag}**: {highlighted_gt}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{tag}**: {gt_val}", unsafe_allow_html=True)
            else:
                st.write(ground_truth)
        
        st.markdown("### Metrics")
        
        # Display Metrics in a Grid
        m_cols = st.columns(len(metrics))
        for idx, (k, v) in enumerate(metrics.items()):
            with m_cols[idx]:
                st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">{k.upper()}</div>
                    <div class="metric-value">{v:.2f}</div>
                </div>
                """, unsafe_allow_html=True)
                
        # --- Visualization (If applicable) ---
        if dataset_name == "screenspot":
            st.markdown("### Visualization")
            pred_coords = parse_coords(generated_text)
            
            # Create a copy to draw on
            vis_image = image.copy()
            draw = ImageDraw.Draw(vis_image)
            w, h = vis_image.size
            
            # Draw Prediction
            if len(pred_coords) >= 2:
                if len(pred_coords) >= 4:
                     # BBox
                    y1, x1, y2, x2 = pred_coords[:4]
                    box = [x1 * w, y1 * h, x2 * w, y2 * h]
                    draw.rectangle(box, outline="red", width=3)
                    draw.text((box[0], box[1]), "Pred", fill="red")
                else:
                    # Point
                    y, x = pred_coords[0], pred_coords[1]
                    px, py = x * w, y * h
                    r = 5
                    draw.ellipse((px-r, py-r, px+r, py+r), fill="red", outline="red")
                    
            # Draw GT if available
            if isinstance(ground_truth, list) and len(ground_truth) == 4:
                xmin, ymin, xmax, ymax = ground_truth
                # GT is usually in pixels for screenspot? Script says: xmin, ymin, xmax, ymax = gt_bbox
                # Wait, the script logic: if xmin <= px <= xmax ... implies GT is in pixels if px is derived from 0-1 coords scaled? 
                # Actually, parse_coords returns 0-1. But in eval_screenspot: px, py = pred_point. 
                # Wait, `eval_screenspot` in script:
                # pred_coords = [0.5, 0.5] (example)
                # px = x.
                # xmin...xmax is GT. 
                # If GT is pixel coordinates, then px (0.5) vs 500 (pixel) comparison would fail unless scaled.
                # Let's check script's eval_screenspot logic carefully.
                # It compares px directly to xmin. 
                # Usually ScreenSpot GT is in 0-1 or pixels?
                # The script `parse_coords` divides by 1024.0. So it normalizes tokens to 0-1.
                # But it finds floats too.
                # If GT is normalized, then it works. If GT is pixels, script might be wrong OR parse_coords handles it.
                # Let's assume GT is compatible with what parse_coords returns (0-1).
                
                # However, drawing requires pixels.
                # If GT is 0-1:
                if max(ground_truth) <= 1.0:
                     draw.rectangle([ground_truth[0]*w, ground_truth[1]*h, ground_truth[2]*w, ground_truth[3]*h], outline="green", width=3)
                else:
                     draw.rectangle(ground_truth, outline="green", width=3)
            
            st.image(vis_image, caption="Red: Prediction, Green: Ground Truth")
            
    else:
        st.info("Click 'Run Inference' to see results.")

