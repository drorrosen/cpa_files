import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
import pandas as pd
import google.generativeai as genai
import faiss
import numpy as np
from pathlib import Path
import json
import openai

# ---------------------------
# Page Configuration
# ---------------------------
st.set_page_config(
    page_title="CPA AI Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------------------------
# Custom CSS Styling - Modern Dark Theme with RTL Support
# ---------------------------
st.markdown("""
<style>
    /* RTL Layout */
    body {
        direction: rtl !important;
    }
    
    .stMarkdown, p, h1, h2, h3, .stChatMessage {
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* Main background - Dark gradient */
    .stApp {
        background: linear-gradient(to bottom left, #0F172A, #1E293B);
    }
    
    /* Text colors and typography */
    .stMarkdown, p {
        color: #E2E8F0 !important;
        line-height: 1.7 !important;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3 {
        background: linear-gradient(120deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    
    /* Chat messages with RTL support */
    .stChatMessage {
        padding: 20px;
        border-radius: 16px;
        margin: 20px 0;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .stChatMessage.user {
        background: rgba(56, 189, 248, 0.1) !important;
        border-right: 4px solid #38BDF8 !important;  /* Changed from left to right */
        border-left: none !important;
        margin-left: 100px !important;  /* Added margin to shift messages */
    }
    
    .stChatMessage.assistant {
        background: rgba(129, 140, 248, 0.1) !important;
        border-right: 4px solid #818CF8 !important;  /* Changed from left to right */
        border-left: none !important;
        margin-right: 100px !important;  /* Added margin to shift messages */
    }
    
    /* Chat input with RTL support */
    .stChatInput {
        background: #FFFFFF !important;
        border: 2px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 12px !important;
        padding: 12px 20px !important;
        color: #1E293B !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1) !important;
        text-align: right !important;
        direction: rtl !important;
    }
    
    .stChatInput:focus {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.3) !important;
    }
    
    .stChatInput::placeholder {
        color: #94A3B8 !important;
    }
    
    /* Style for the input text */
    .stChatInput input {
        color: #1E293B !important;
        font-size: 1.1em !important;
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* Success message */
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border-left: 4px solid #10B981 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    
    /* Error message */
    .stError {
        background: rgba(239, 68, 68, 0.2) !important;
        border-left: 4px solid #EF4444 !important;
        color: #E2E8F0 !important;
        border-radius: 8px !important;
        padding: 20px !important;
    }
    
    /* Header container with centered title */
    .header-container {
        background: rgba(30, 41, 59, 0.6);
        padding: 40px;
        border-radius: 24px;
        margin: 20px 0 40px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        text-align: center !important;  /* Center the header content */
    }
    
    .header-container h1, .header-container p {
        text-align: center !important;  /* Force center alignment for header text */
        direction: rtl !important;
    }
    
    /* Footer */
    footer {
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
        color: #64748B !important;
        font-size: 0.9em;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(30, 41, 59, 0.2);
    }
    
    ::-webkit-scrollbar-thumb {
        background: #38BDF8;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #818CF8;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Header with centered title
# ---------------------------
st.markdown("""
<div class="header-container">
    <h1 style="text-align: center; margin-bottom: 15px; font-size: 2.5em;">🤖 CPA AI Assistant</h1>
    <p style="text-align: center; font-size: 1.2em; color: #94A3B8;">Unlock the power of your documents with AI</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# API Key Setup
# ---------------------------
# TODO: Move these to st.secrets in production
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]


# Add Gemini import
import google.generativeai as genai

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Create Gemini model configuration
generation_config = {
    "temperature": 0.0,
    "top_p": 0.5,
    "top_k": 20,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize Gemini model
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.0-pro-exp-02-05",
    generation_config=generation_config,
)

@st.cache_resource
def initialize_embeddings():
    return OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"]
)

def initialize_vector_store():
    """Initialize or rebuild vector store from scratch each time"""
    print("Starting index building process...")
    
    # Initialize OpenAI embeddings
    embeddings_model = OpenAIEmbeddings(openai_api_key=st.secrets["OPENAI_API_KEY"])
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200000,
        chunk_overlap=50000,
        separators=["\n\n=== Document:", "\n\n", "\n", " ", ""]
    )
    
    # Process all text files
    texts_dir = Path("extracted_texts")
    texts = []
    metadatas = []
    
    print("Loading documents...")
    for text_file in texts_dir.glob("*.txt"):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks
            chunks = text_splitter.split_text(content)
            texts.extend(chunks)
            
            # Add metadata for each chunk
            for chunk_idx in range(len(chunks)):
                metadatas.append({
                    "filename": text_file.name,
                    "chunk_index": chunk_idx,
                    "total_chunks": len(chunks)
                })
            
        except Exception as e:
            print(f"Error processing file {text_file.name}: {str(e)}")
            continue
    
    print(f"Creating FAISS index from {len(texts)} text chunks...")
    
    # Create FAISS index from texts
    vector_store = FAISS.from_texts(
        texts=texts,
        embedding=embeddings_model,
        metadatas=metadatas
    )
    
    return vector_store

# Add this near the top of your app, after imports
if 'vector_store' not in st.session_state:
    try:
        with st.status("Building vector store..."):
            st.session_state.vector_store = initialize_vector_store()
            st.success("Vector store built successfully!")
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")

# Use the stored vector store for queries
vector_store = st.session_state.vector_store

# ---------------------------
# Define helper functions first
# ---------------------------
def display_formatted_response(answer):
    # Check if the response contains a table
    if "|" in answer and "-|-" in answer:
        # Split the response into explanation and table parts
        parts = answer.split("\n\n")
        explanation = parts[0]
        table_text = "\n".join([p for p in parts if "|" in p])
        
        # Display the explanation
        st.markdown(explanation)
        
        # Convert markdown table to DataFrame
        try:
            # Parse the markdown table
            lines = [line.strip() for line in table_text.split('\n') if line.strip()]
            headers = [col.strip() for col in lines[0].split('|') if col.strip()]
            data = []
            for line in lines[2:]:  # Skip the separator line
                row = [cell.strip() for cell in line.split('|') if cell.strip()]
                data.append(row)
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=headers)
            
            # Display as a styled table
            st.markdown("""
                <style>
                    .dataframe {
                        direction: rtl;
                        font-family: 'Arial', sans-serif;
                        width: 100%;
                        text-align: right;
                    }
                    .dataframe th {
                        background-color: #1a365d;
                        color: white;
                        padding: 12px;
                        text-align: right !important;
                    }
                    .dataframe td {
                        background-color: rgba(255, 255, 255, 0.05);
                        padding: 8px;
                        text-align: right !important;
                    }
                    .dataframe tr:hover td {
                        background-color: rgba(255, 255, 255, 0.1);
                    }
                </style>
            """, unsafe_allow_html=True)
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.markdown(answer)  # Fallback to regular markdown if table parsing fails
    else:
        st.markdown(answer)

def get_specialized_prompt(question, context):
    # Add document source information to the prompt
    if any(word in question.lower() for word in ['מאזן', 'טבלה', 'השוואה', 'נתונים']):
        return f"""
        You are a financial data analyst. Create a clear, structured response:

        1. First, identify which document(s) contain the relevant information
        2. Then provide a brief explanation
        3. Present the data in a format that can be converted to a pandas DataFrame
        4. Use this exact format for tables:

        Column1 | Column2 | Column3
        --------|---------|--------
        Value1  | Value2  | Value3

        Rules:
        - Mention the source document(s) in your explanation
        - Use Hebrew column names
        - Align numbers to the right
        - Format numbers with commas
        - Show percentages with % symbol
        - Use clear section headers
        
        Context: {context}
        Question: {question}
        """
    else:
        return f"""
        Format your response with:
        1. Mention which document(s) contain the information
        2. Numbered points for lists
        3. Bold for important values
        4. Clear sections if needed
        5. In the hebrew language and organize it.
        
        Context: {context}
        Question: {question}
        
        Make the answer clear and well-structured.
        """

# ---------------------------
# Chat Interface
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        relevant_docs = vector_store.similarity_search(prompt, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        full_prompt = get_specialized_prompt(prompt, context)
        
        # Comment out the old OpenAI model
        # chat_model = ChatOpenAI(
        #     model_name="gpt-4o",
        #     openai_api_key=OPENAI_API_KEY,
        #     max_tokens=10000
        # )
        # response = chat_model.invoke(full_prompt)
        # answer = response.content
        
        # Use Gemini model instead
        chat_session = gemini_model.start_chat(history=[])
        response = chat_session.send_message(full_prompt)
        answer = response.text
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            display_formatted_response(answer)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add a footer
st.markdown("---")
st.markdown("*Powered by OpenAI and FAISS*") 
