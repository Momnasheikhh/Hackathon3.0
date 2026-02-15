import google.generativeai as genai
import pandas as pd
import json
import os

# Configure Gemini
# Ideally, this should be called after loading env vars
def configure_genai():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        api_key = "AIzaSyDQ0OCcqyQX15qMRcQMWDGFveIbGR3RLYU" # Fallback from user's file (temporary)
    genai.configure(api_key=api_key)

def get_model():
    configure_genai()
    # Using the model specified in the user's file
    return genai.GenerativeModel("gemini-2.5-flash-lite")

def parse_with_llm(raw_text):
    model = get_model()
    prompt = f"""
    You are a highly advanced AI trained specifically for Receipt Data Extraction and Financial Analysis.
    
    ### INPUT DATA (OCR Text):
    {raw_text}
    
    ### YOUR MISSION:
    Accurately extract the purchase items from the receipt text.
    
    ### STRICT RULES (CRITICAL):
    1. **NO SUMMARY LINES**: ABSOLUTELY IGNORE lines with "Total", "Subtotal", "Tax", "VAT", "Balance", "Change", "Cash", "Credit", "Amount Due".
    2. **NO NOISE**:
        - Ignore Phone Numbers (e.g., 555-0199, +1-234...).
        - Ignore Dates (e.g., 01/01/2018).
        - Ignore random codes or barcodes.
    3. **PRICE VALIDATION**:
        - Prices are usually between 0.10 and 500.00 for normal items.
        - If a number is huge (e.g., 10002018), it is likely an ID or Phone Number -> IGNORE IT.
    4. **DATA TYPES**:
       - "item": String (Clean product name).
       - "price": Float (Numeric only, remove currency symbols).
       - "quantity": Integer (Default 1).
       - "category": String (Categorize properly: "Food", "Grocery", "Shopping", "Transport", "Utility", "Health", "Other").
    
    ### OUTPUT FORMAT:
    Return **ONLY** a raw JSON array.
    
    ### EXAMPLE:
    [
      {{"item": "Cappuccino", "price": 4.50, "quantity": 1, "category": "Food"}},
      {{"item": "Notebook", "price": 2.99, "quantity": 2, "category": "Shopping"}}
    ]
    """
    
    text = "Error: No response"
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Cleanup markdown formatting common in LLM responses
        clean_text = text.replace("```json", "").replace("```", "").strip()
        
        # Try finding the list start/end if there's extra text
        start_idx = clean_text.find('[')
        end_idx = clean_text.rfind(']')
        
        if start_idx != -1 and end_idx != -1:
            clean_text = clean_text[start_idx:end_idx+1]
        
        return json.loads(clean_text)
        
    except Exception as e:
        # RETURN RAW TEXT FOR DEBUGGING if parsing fails
        print(f"LLM JSON Parsing Failed. Raw response: {text}") 
        return []

def analyze_spending(data):
    if not data:
        return None, "No data to analyze."
    
    df = pd.DataFrame(data)
    
    # Check if 'price' column requires cleaning (removing currency symbols)
    if 'price' in df.columns and df['price'].dtype == 'object':
         df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
         df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)

    total = df["price"].sum()
    category_breakdown = df.groupby("category")["price"].sum().to_dict()
    
    return df, {"total": total, "breakdown": category_breakdown}

def get_ai_advice(df):
    model = get_model()
    prompt = f"""
    You are a financial advisor.
    
    Spending data:
    {df.to_dict(orient='records')}
    
    Give short saving advice.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error getting advice: {e}"
