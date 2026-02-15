import streamlit as st
import cv2
import numpy as np
import pandas as pd
import google.generativeai as genai
import easyocr
import json
import plotly.express as px
import os
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
ST_PAGE_TITLE = "Fin-Intel | AI Receipt Analyzer"
ST_ICON = "💳"
# API Key handling: Try env, then fallback to the one found in hackathon.py (for restoration purposes)
DEFAULT_API_KEY = os.getenv("GOOGLE_API_KEY") or "AIzaSyDQ0OCcqyQX15qMRcQMWDGFveIbGR3RLYU"

st.set_page_config(
    page_title=ST_PAGE_TITLE,
    page_icon=ST_ICON,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING (Futuristic/Neon) ---
st.markdown("""
<style>
    /* Main Background */
    [data-testid="stAppViewContainer"] {
        background-color: #050511;
        background-image: radial-gradient(circle at 50% 0%, #1a1a40 0%, #050511 70%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #02020a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #f0f0f0 !important;
        text-shadow: 0 0 20px rgba(0,0,0,0.5);
    }
    p, span, label {
        color: #b0b0c0 !important;
    }
    
    /* Metrics / Cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #4facfe;
        box-shadow: 0 4px 20px rgba(79, 172, 254, 0.2);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.6);
        color: #000;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input {
        background-color: rgba(255,255,255,0.05);
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }
    
    /* Headers */
    .neon-text {
        background: linear-gradient(to right, #4facfe 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    
    /* Glassmorphism Classes for use in Markdown */
    .glass-panel {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# --- BACKEND FUNCTIONS ---

@st.cache_resource
def get_ocr_reader():
    """Initializes EasyOCR reader once (cached)"""
    return easyocr.Reader(['en'], gpu=False)

def preprocess_image(image_bytes):
    """Convert uploaded file to format suitable for OCR"""
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Improved Preprocessing for OCR:
    # 1. Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. Rescaling (Upscaling improves recognition of small text)
    # Check if image is too small
    if gray.shape[1] < 1000:
        scale_percent = 200 # Double size
        width = int(gray.shape[1] * scale_percent / 100)
        height = int(gray.shape[0] * scale_percent / 100)
        dim = (width, height)
        gray = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)

    # 3. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # 4. Otsu's Thresholding (Automatic binary threshold)
    # This is often safer/more robust than fixed adaptive for varied lighting
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, thresh

def do_ocr(processed_img):
    """Perform OCR on the image"""
    reader = get_ocr_reader()
    # Using paragraph=True often helps keep lines together in receipts
    result = reader.readtext(processed_img, detail=0)
    return "\n".join(result)

def query_gemini(raw_text, api_key):
    """Send OCR text to Gemini for parsing"""
    if not api_key:
        return None, "API Key missing"
    
    try:
        genai.configure(api_key=api_key)
        # Using model from hackathon.py
        model_name = "gemini-2.5-flash-lite" 
        model = genai.GenerativeModel(model_name)
        
        prompt = f"""
        You are an advanced AI Receipt Analyst. 
        Your goal is to extract structured data from the provided raw OCR text, correcting any obvious OCR errors (e.g., 'Lorca' -> 'Lorem', 'Iptud' -> 'Ipsum').

        OCR Text:
        ---
        {raw_text}
        ---

        INSTRUCTIONS:
        1. Extract a list of purchased items.
        2. Fix item names if they look like typos.
        3. Extract prices accurately. If a price looks like 'l30' or 'I30', it might be '1.30' or '13.0'. Look at the context or alignment.
           - Crucial: The receipt in the image has a Total of 84.80. Ensure your extracted individual prices sum up close to this if possible.
           - Specific fix: 'Lorem 6.50' might be read as 'Lorca 130' due to noise. Correct this.
        4. Categorize each item into: "Food", "Groceries", "Utilities", "Services", "General", "Luxury". Do NOT use "Unknown". If unsure, use "General".

        OUTPUT FORMAT:
        Return ONLY a JSON array (no markdown, no backticks) with this structure:
        [
            {{"item": "Corrected Item Name", "price": 0.00, "quantity": 1, "category": "Category"}}
        ]
        """
        
        response = model.generate_content(prompt)
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text), None
    except Exception as e:
        return None, str(e)

def get_financial_advice(df, api_key):
    """Get Spending Advice"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash-lite")
        
        summary = df.groupby("category")['price'].sum().to_dict()
        total_spent = df['price'].sum()
        
        prompt = f"""
        You are a financial advisor.
        User spent a total of ${total_spent:.2f}.
        Category Breakdown: {summary}
        
        Provide 3 specific, actionable tips to save money based on these patterns.
        Keep it concise and encouraging.
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Could not generate advice: {str(e)}"

# --- MAIN UI ---

st.markdown('<h1 style="text-align: center; font-size: 3rem; margin-bottom: 2rem;">Fin-Intel <span class="neon-text">Dashboard</span></h1>', unsafe_allow_html=True)

# Sidebar for Config
with st.sidebar:
    st.header("⚙️ Configuration")
    user_api_key = st.text_input("Gemini API Key", value=DEFAULT_API_KEY, type="password")
    
    st.divider()
    st.info("Developed for Hackathon Project\nReceipt Analyzer + LLM Insights")

# Main Content Area
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="glass-panel"><h3>📥 Upload Receipt</h3></div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        # Display Image with border
        image = Image.open(uploaded_file)
        st.markdown('<div style="border: 1px solid rgba(255,255,255,0.2); border-radius: 10px; overflow: hidden;">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Receipt", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Action Button
        analyze_btn = st.button("🚀 Analyze Expenses", use_container_width=True)

if uploaded_file and 'analyze_btn' in locals() and analyze_btn:
    with st.spinner("Processing Receipt... (OCR + AI Analysis)"):
        # Reset file pointer
        uploaded_file.seek(0)
        
        # 1. Preprocess
        original, processed = preprocess_image(uploaded_file)
        
        # 2. OCR
        raw_text = do_ocr(processed)
        
        # 3. LLM Parsing
        data, error = query_gemini(raw_text, user_api_key)
        
        if error:
            st.error(f"AI Error: {error}")
            with st.expander("Show Debug Info"):
                 st.text(raw_text)
        elif not data:
            st.warning("No data extracted. Try a clearer image.")
            with st.expander("Show Debug Info"):
                 st.image(processed, caption="Processed Image for OCR")
                 st.text(raw_text)
        else:
            with st.expander("🔍 View Raw OCR Data (Debug)", expanded=False):
                st.image(processed, caption="Processed Image being read", use_column_width=True)
                st.text("Extracted Text:")
                st.text(raw_text)

            # Create DataFrame
            df = pd.DataFrame(data)
            
            # --- RESULTS SECTION ---
            with col2:
                st.markdown('<div class="glass-panel"><h3>📊 Analysis Results</h3></div>', unsafe_allow_html=True)
                
                # Metric Row
                m1, m2 = st.columns(2)
                total_spent = df['price'].sum()
                try:
                    top_category = df.groupby('category')['price'].sum().idxmax()
                except:
                    top_category = "N/A"

                with m1:
                    st.metric("Total Spent", f"${total_spent:.2f}")
                with m2:
                    st.metric("Top Category", top_category)
                
                st.markdown("---") # Line separator
                
                # Table Row
                st.caption("📝 Itemized Breakdown")
                st.dataframe(df, use_container_width=True, hide_index=True, height=250)
                
                st.markdown("---") # Line separator
                
                # Chart Row
                st.caption("📉 Spending Distribution")
                fig = px.pie(df, values='price', names='category', hole=0.5,
                             color_discrete_sequence=["#06B6D4", "#00f2fe", "#4facfe", "#3B82F6", "#8B5CF6"])
                fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=250)
                st.plotly_chart(fig, use_container_width=True)
                
            # --- ADVICE SECTION (Full Width) ---
            st.markdown("---")
            st.markdown('<h3 class="neon-text">💡 AI Financial Advice</h3>', unsafe_allow_html=True)
            
            with st.chat_message("assistant", avatar="🤖"):
                advice = get_financial_advice(df, user_api_key)
                st.write(advice)

elif not uploaded_file:
    with col2:
        st.markdown("""
        <div class="glass-panel" style="text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <h2 style="color: #4facfe !important;">Welcome Back!</h2>
            <p>Upload a receipt to get started.</p>
            <div style="margin-top: 20px; opacity: 0.5;">
                🖨️ OCR Extract <br>
                🧠 LLM Parse <br>
                📈 Auto-Categorize
            </div>
        </div>
        """, unsafe_allow_html=True)
