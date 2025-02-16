import faiss
import numpy as np
import json
from pathlib import Path
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Change to use LangChain

def build_index():
    print("Starting index building process...")
    
    # Initialize
    dimension = 3072
    index = faiss.IndexFlatL2(dimension)
    metadata = {}
    
    # Initialize OpenAI embeddings through LangChain
    embeddings_model = OpenAIEmbeddings(
        openai_api_key="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"
    )
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200000,  # Can use larger chunks now
        chunk_overlap=50000,
        separators=["\n\n=== Document:", "\n\n", "\n", " ", ""]
    )
    
    # Create faiss_index directory if it doesn't exist
    os.makedirs("faiss_index", exist_ok=True)
    
    # Process all text files
    texts_dir = Path("extracted_texts")
    total_files = len(list(texts_dir.glob("*.txt")))
    
    print(f"Found {total_files} text files to process")
    chunk_counter = 0
    
    for i, text_file in enumerate(texts_dir.glob("*.txt"), 1):
        print(f"Processing file {i}/{total_files}: {text_file.name}")
        
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split content into chunks
            chunks = text_splitter.split_text(content)
            print(f"Split {text_file.name} into {len(chunks)} chunks")
            
            for chunk_idx, chunk in enumerate(chunks):
                try:
                    # Get embeddings for each chunk using LangChain
                    print(f"Getting embedding for chunk {chunk_idx + 1}/{len(chunks)}")
                    embedding = embeddings_model.embed_query(chunk)  # Changed to use LangChain
                    
                    # Add to index
                    index.add(np.array([embedding]).astype('float32'))
                    
                    # Store metadata
                    metadata[str(chunk_counter)] = {
                        "filename": text_file.name,
                        "chunk_index": chunk_idx,
                        "content": chunk,
                        "total_chunks": len(chunks)
                    }
                    chunk_counter += 1
                    
                except Exception as e:
                    print(f"Error processing chunk {chunk_idx} of {text_file.name}: {str(e)}")
                    continue
            
            print(f"Successfully processed all chunks of {text_file.name}")
            
        except Exception as e:
            print(f"Error processing file {text_file.name}: {str(e)}")
            continue
    
    print("\nSaving index and metadata...")
    
    # Save everything
    faiss.write_index(index, "faiss_index/document_index.faiss")
    with open("faiss_index/metadata.json", 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False)
    
    print("\nIndex building complete!")
    print(f"Processed {len(metadata)} chunks successfully")
    print("You can now deploy to Streamlit!")

if __name__ == "__main__":
    build_index() 