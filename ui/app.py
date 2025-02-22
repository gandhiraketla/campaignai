import streamlit as st
import requests
from PIL import Image
import io

# Page Configuration
st.set_page_config(
    page_title="CampaignAI",
    page_icon="🚀",
    layout="wide"
)

# Custom CSS for professional look
st.markdown("""
<style>
    .main-header {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header img {
        max-height: 60px;
        margin-right: 20px;
    }
    .main-header h1 {
        color: #2c3e50;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        background-color: #f8f9fa;
        border: 1px solid #ced4da;
        border-radius: 8px;
        padding: 10px;
        font-size: 1rem;
    }
    .stButton > button {
        background-color: #3498db;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 1rem;
        transition: background-color 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .response-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Generate Logo (Create a simple SVG logo)
logo_svg = '''
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
    <rect width="100" height="100" rx="20" fill="#3498db"/>
    <text x="50%" y="50%" text-anchor="middle" dy=".3em" 
        font-family="Arial, sans-serif" 
        font-size="40" 
        font-weight="bold" 
        fill="white">CA</text>
</svg>
'''

# Header
st.markdown("""
<div class="main-header">
    <div style="display: flex; align-items: center;">
        {logo}
        <h1>CampaignAI</h1>
    </div>
</div>
""".format(logo=logo_svg), unsafe_allow_html=True)

# Main Application
def main():
    # Query Input
    st.markdown("<h2 style='text-align: center; color: #2c3e50;'>AI-Powered Campaign Insights</h2>", unsafe_allow_html=True)
    
    # Create columns for better layout
    col1, col2, col3 = st.columns([1,3,1])
    
    with col2:
        # User Query Input
        user_query = st.text_input(
            "Enter your campaign or marketing query", 
            placeholder="e.g., How are our Facebook campaigns performing this month?"
        )
        
        # Submit Button
        submit_button = st.button("Get Insights", use_container_width=True)
        
        # Response Handling
        if submit_button and user_query:
            try:
                # Replace with your actual API endpoint
                response = requests.post(
                    "http://localhost:8000/query", 
                    json={"query": user_query},
                    timeout=30
                )
                
                if response.status_code == 200:
                    # Display Response
                    st.markdown("<div class='response-box'>", unsafe_allow_html=True)
                    st.markdown(f"**CampaignAI Insights:**")
                    st.write(response.json()['response'])
                    st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")
            
            except requests.RequestException as e:
                st.error(f"Connection Error: {e}")
                st.error("Unable to connect to the CampaignAI service. Please check the API is running.")

# Footer
st.markdown("""
<div style='text-align: center; color: #7f8c8d; margin-top: 50px;'>
    <p>© 2024 CampaignAI. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()