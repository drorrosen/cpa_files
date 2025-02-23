import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
import google.generativeai as genai
from typing import List
from langchain.docstore.document import Document
import pandas as pd

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="CPA AI Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# Enhanced Custom CSS Styling
# ---------------------------
st.markdown("""
<style>
    /* RTL Layout */
    body { 
        direction: rtl !important; 
        font-family: 'Heebo', sans-serif;
    }
    
    /* Main background - Enhanced gradient */
    .stApp {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    }
    
    /* Text and Typography */
    .stMarkdown, p, h1, h2, h3, .stChatMessage {
        text-align: right !important;
        direction: rtl !important;
        color: #E2E8F0 !important;
        line-height: 1.7 !important;
    }
    
    h1, h2, h3 {
        background: linear-gradient(120deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* Enhanced Chat Messages */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
    }
    
    .stChatMessage:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 0, 0, 0.3);
    }
    
    .stChatMessage.user {
        background: rgba(56, 189, 248, 0.1) !important;
        border-right: 4px solid #38BDF8 !important;
        margin-left: 100px !important;
    }
    
    .stChatMessage.assistant {
        background: rgba(129, 140, 248, 0.1) !important;
        border-right: 4px solid #818CF8 !important;
        margin-right: 100px !important;
    }
    
    /* Enhanced Chat Input */
    .stTextInput > div > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        border: 2px solid rgba(56, 189, 248, 0.3) !important;
        padding: 0.75rem !important;
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div:hover {
        border-color: #38BDF8 !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    .stTextInput > div > div:focus-within {
        border-color: #818CF8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2) !important;
    }
    
    /* Tables */
    .dataframe {
        direction: rtl;
        width: 100%;
        margin: 1.5rem 0;
        border-collapse: separate;
        border-spacing: 0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    .dataframe th {
        background: linear-gradient(90deg, #1a365d, #2d4a7c);
        color: white;
        padding: 1rem;
        text-align: right !important;
        font-weight: 600;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
    }
    
    .dataframe td {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 0.75rem;
        text-align: right !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .dataframe tr:last-child td {
        border-bottom: none;
    }
    
    .dataframe tr:hover td {
        background-color: rgba(255, 255, 255, 0.08);
        transition: background-color 0.3s ease;
    }
    
    /* Source Citations */
    .source-citation {
        background: rgba(56, 189, 248, 0.1);
        border-right: 3px solid #38BDF8;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-size: 0.9em;
    }
    
    /* Status Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #38BDF8, #818CF8);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #818CF8, #38BDF8);
    }
    
    /* Header Container */
    .header-container {
        background: rgba(30, 41, 59, 0.6);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 1.5rem 0 3rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
        color: #64748B;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Enhanced Header
# ---------------------------
st.markdown("""
<div class="header-container">
    <h1>🤖 CPA AI Assistant</h1>
    <p style="font-size: 1.2em; color: #94A3B8; margin-top: 1rem;">חכמת המסמכים הפיננסיים שלך</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# API Key Setup from Secrets
# ---------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = "us-east-1"

# Set Pinecone environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

# ---------------------------
# Initialize Services
# ---------------------------
@st.cache_resource
def initialize_gemini():
    """Initialize Gemini model"""
    genai.configure(api_key=GEMINI_API_KEY)
    generation_config = {
        "temperature": 0.0,
        "top_p": 0.5,
        "top_k": 20,
        "max_output_tokens": 8192,
    }
    return genai.GenerativeModel(
        model_name="gemini-2.0-pro-exp-02-05",
        generation_config=generation_config,
    )

@st.cache_resource
def initialize_vector_store():
    """Initialize Pinecone vector store"""
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    
    return Pinecone.from_existing_index(
        index_name="index",
        embedding=embeddings_model,
        namespace="Default"
    )

def get_relevant_documents(vector_store, query: str, score_threshold: float = 0.7, k: int = 5) -> List[Document]:
    """Get relevant documents with similarity scores above threshold"""
    results = vector_store.similarity_search_with_score(query, k=k)
    
    # Filter results above threshold and sort by score
    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]
    filtered_results.sort(key=lambda x: x[1], reverse=True)
    
    # Return top 3 matches
    return [doc for doc, _ in filtered_results[:3]]

def format_source_info(doc: Document) -> str:
    """Format document source information"""
    source = doc.metadata.get('source', 'Unknown source')
    page = doc.metadata.get('page', 'N/A')
    return f"מסמך: {source} (עמוד {page})"

def get_specialized_prompt(question: str, context: str, doc_sources: List[str]) -> str:
    """Create specialized prompt with document sources"""
    sources_text = "\n".join([f"- {source}" for source in doc_sources])
    
    if any(word in question.lower() for word in ['מאזן', 'טבלה', 'דוח', 'נתונים']):
        return f"""
        You are a financial data analyst. Create a clear, structured response in Hebrew.
        
        Available source documents:
        {sources_text}
        
        Rules for your response:
        1. Start by mentioning which specific documents you're using
        2. Present financial data in clear, RTL-formatted tables
        3. Use Hebrew column names
        4. Format numbers with commas and right alignment
        5. Include % symbol for percentages
        6. Add explanatory text before and after tables
        
        Context: {context}
        Question: {question}
        """
    else:
        return f"""
        You are a financial expert assistant. Respond in Hebrew with:
        
        Available source documents:
        {sources_text}
        
        Format your response with:
        1. Cite which specific documents you're using
        2. Use bullet points for lists
        3. Bold for key figures and important points
        4. Organize information in clear sections
        5. Maintain right-to-left (RTL) formatting
        
        Context: {context}
        Question: {question}
        """

def display_formatted_response(answer: str, sources: List[str] = None):
    """Display the formatted response with enhanced styling"""
    if sources:
        st.markdown("**:blue[מקורות המידע:]**")
        for source in sources:
            st.markdown(f"""
                <div class="source-citation">
                    📄 {source}
                </div>
            """, unsafe_allow_html=True)
        st.markdown("---")
    
    if "|" in answer and "-|-" in answer:
        parts = answer.split("\n\n")
        for part in parts:
            if "|" in part and "-|-" in part:
                try:
                    lines = [line.strip() for line in part.split('\n') if line.strip()]
                    headers = [col.strip() for col in lines[0].split('|') if col.strip()]
                    data = []
                    for line in lines[2:]:
                        row = [cell.strip() for cell in line.split('|') if cell.strip()]
                        data.append(row)
                    df = pd.DataFrame(data, columns=headers)
                    st.dataframe(df, use_container_width=True)
                except:
                    st.markdown(part)
            else:
                st.markdown(part)
    else:
        st.markdown(answer)

# ---------------------------
# Main Chat Interface
# ---------------------------
try:
    gemini_model = initialize_gemini()
    vector_store = initialize_vector_store()
except Exception as e:
    st.error(f"שגיאה באתחול השירותים: {str(e)}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sources" in message:
            display_formatted_response(message["content"], message["sources"])
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("שאל שאלה על המסמכים שלך..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Get relevant documents with scores
        relevant_docs = get_relevant_documents(vector_store, prompt)
        
        if not relevant_docs:
            response_text = "לא נמצאו מסמכים רלוונטיים מספיק לשאלה שלך. אנא נסה לנסח את השאלה אחרת."
            sources = []
        else:
            # Extract document sources and content
            sources = [format_source_info(doc) for doc in relevant_docs]
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            
            # Generate response
            full_prompt = get_specialized_prompt(prompt, context, sources)
            response = gemini_model.generate_content(full_prompt)
            response_text = response.text

        # Add assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "sources": sources
        })
        
        with st.chat_message("assistant"):
            display_formatted_response(response_text, sources)

    except Exception as e:
        st.error(f"שגיאה: {str(e)}")

# Enhanced Footer
st.markdown("""
<div class="footer">
    <p>Powered by Google Gemini and Pinecone</p>
    <p style="margin-top: 0.5rem;">© 2024 CPA AI Assistant</p>
</div>
""", unsafe_allow_html=True) 
