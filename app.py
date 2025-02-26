import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import Pinecone
import anthropic
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
    
    /* Sidebar Styling */
    .sidebar-content {
        background: rgba(0, 0, 0, 0.8) !important;
        padding: 20px !important;
        border-radius: 10px !important;
        margin: 10px !important;
    }
    
    /* Sidebar text */
    .sidebar .sidebar-content [data-testid="stMarkdownContainer"] {
        color: white !important;
    }
    
    /* Sidebar button */
    .stButton > button {
        background-color: #38BDF8 !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        width: 100% !important;
        margin-bottom: 20px !important;
    }
    
    /* Document list items */
    .document-item {
        background: rgba(255, 255, 255, 0.9);
        padding: 12px 15px;
        border-radius: 8px;
        margin: 8px 0;
        border-right: 4px solid #38BDF8;
        font-size: 1.1em;
        color: #000000 !important;
        font-weight: 500;
        text-shadow: none;
        display: block;
        transition: all 0.2s ease;
    }

    .document-item:hover {
        background: rgba(255, 255, 255, 1);
        transform: translateX(-5px);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar header */
    .sidebar-header {
        color: #38BDF8 !important;
        font-size: 1.3em !important;
        font-weight: bold !important;
        margin-bottom: 15px !important;
        text-align: center !important;
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
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = "us-east-1"

# Set Pinecone environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT

# ---------------------------
# Initialize Services
# ---------------------------
@st.cache_resource
def initialize_claude():
    """Initialize Claude model"""
    return anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
    )

@st.cache_resource
def initialize_vector_store():
    """Initialize Pinecone vector store"""
    embeddings_model = OpenAIEmbeddings(
        openai_api_key=OPENAI_API_KEY,
        model="text-embedding-3-large"
    )
    
    # Initialize Pinecone with new SDK syntax
    from pinecone import Pinecone
    from langchain_pinecone import PineconeVectorStore
    
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Get the index
    index = pc.Index(
        "index",
        host="index-fmrj1el.svc.aped-4627-b74a.pinecone.io"
    )
    
    return PineconeVectorStore.from_existing_index(
        index_name="index",
        embedding=embeddings_model,
        namespace="Default"
    )

@st.cache_data
def get_available_documents():
    """Get list of unique documents in the vector store"""
    try:
        # Query with empty string to get sample of vectors
        results = vector_store.similarity_search_with_score("", k=1000)
        # Extract unique filenames
        unique_docs = set()
        for doc, _ in results:
            # Check both original_file and file fields
            if 'original_file' in doc.metadata:
                unique_docs.add(doc.metadata['original_file'])
            elif 'file' in doc.metadata:
                # Decode the encoded filename
                encoded_filename = doc.metadata['file']
                if '_' in encoded_filename:
                    _, original = encoded_filename.split('_', 1)
                    unique_docs.add(original)
                else:
                    unique_docs.add(encoded_filename)
        return sorted(list(unique_docs))
    except Exception as e:
        st.sidebar.error(f"Error fetching documents: {str(e)}")
        return []

def get_relevant_documents(vector_store, query: str, score_threshold: float = 0.25, k: int = 5) -> List[Document]:
    """Get relevant documents with similarity scores above threshold"""
    try:
        st.sidebar.write("🔍 Searching for documents...")
        results = vector_store.similarity_search_with_score(query, k=k)
        
        if not results:
            st.sidebar.warning("No results found in initial search")
            return []
            
        # Debug print all scores
        st.sidebar.write("📊 Raw Results:")
        for doc, score in results:
            filename = doc.metadata.get('original_file') or doc.metadata.get('file', 'Unknown')
            if '_' in filename and 'original_file' not in doc.metadata:
                _, filename = filename.split('_', 1)
            st.sidebar.write(f"Score: {score:.3f} - {filename}")
        
        # Filter with lower threshold
        filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        st.sidebar.write(f"🎯 Filtered results (threshold {score_threshold}):")
        st.sidebar.write(f"Found {len(filtered_results)} relevant documents")
        
        if not filtered_results:
            st.sidebar.warning("No documents passed the relevance threshold")
            # Return top result anyway if available
            if results:
                st.sidebar.info("Returning top result despite low score")
                return [results[0][0]]
            return []
            
        return [doc for doc, _ in filtered_results[:5]]
    except Exception as e:
        st.sidebar.error(f"Error in get_relevant_documents: {str(e)}")
        return []

def format_source_info(doc: Document) -> str:
    """Format document source information"""
    filename = doc.metadata.get('original_file')
    if not filename:
        filename = doc.metadata.get('file', 'Unknown source')
        if '_' in filename:
            _, filename = filename.split('_', 1)
    
    page = doc.metadata.get('chunk', 'N/A')
    total_chunks = doc.metadata.get('total_chunks', '?')
    return f"מסמך: {filename} (חלק {page}/{total_chunks})"

def get_specialized_prompt(question: str, context: str, doc_sources: List[str]) -> str:
    """Create specialized prompt with document sources"""
    sources_text = "\n".join([f"- {source}" for source in doc_sources])
    
    # New standardized prompt in Hebrew
    return f"""
    אתה מומחה לדיווחים כספיים ועוזר עסקי. המשימה שלך היא לקרוא, לנתח ולהסביר את תוכן הדוחות הכספיים של חברות בישראל. הדוחות כוללים את החלקים הבאים:

    • *תיאור עסקי:* סקירה מקיפה של פעילות התאגיד לשנה המדוברת.
    • *דוח דירקטוריון:* דיווח על מצב עסקי החברה, הכולל:
       - חלק א': הסברים מהדירקטוריון למצב העסקי.
       - חלק ב': חשיפה לסיכוני שוק ודרכי ניהולם.
       - חלק ג': היבטי מימשל תאגידי.
       - חלק ד': הוראות גילוי והצהרות הקשורות לדיווח הפיננסי.
    • *דוחות כספיים:*
       - דוח סקירה של רואה החשבון.
       - דוחות מאוחדים הכוללים: מצב כספי, דוח רווח/הפסד, דוח שינויים בהון, ודוח תזרימי מזומנים.
       - באורים לדוחות הכספיים.
       - דוח על אפקטיביות הבקרה הפנימית על הדיווח הכספי.

    עליך להציג את המידע בצורה ברורה ומסודרת, תוך שימוש בטבלאות (כאשר יש נתונים מספריים), הדגשה של נתונים מרכזיים והסברים מפורטים בשפה העברית עם עיצוב תומך RTL. כמו כן, ציין מהם המקורות מהם שאבת את המידע.

    השאלות שיתבקשו ידרשו ניתוח מעמיק של הנתונים, השוואות בין תקופות ודיווח על מסקנות עיקריות בהתאם למידע בדוחות.
    
    מקורות זמינים:
    {sources_text}
    
    הקשר: {context}
    
    שאלה: {question}
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
    claude_client = initialize_claude()
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

# Add after the header and before the chat interface
if st.sidebar.button("הצג מסמכים זמינים"):
    documents = get_available_documents()
    if isinstance(documents, list):
        st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="sidebar-header">מסמכים זמינים</div>', unsafe_allow_html=True)
        for doc in documents:
            st.sidebar.markdown(f'<div class="document-item">📑 {doc}</div>', unsafe_allow_html=True)
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    else:
        st.sidebar.error(documents)

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
            
            # Generate response using Claude
            full_prompt = get_specialized_prompt(prompt, context, sources)
            
            # Create Claude message
            response = claude_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                temperature=0,
                max_tokens=4096,
                messages=[
                    {"role": "user", "content": full_prompt}
                ]
            )
            response_text = response.content[0].text

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
    <p>Powered by Anthropic Claude and Pinecone</p>
    <p style="margin-top: 0.5rem;">© 2024 CPA AI Assistant</p>
</div>
""", unsafe_allow_html=True) 
