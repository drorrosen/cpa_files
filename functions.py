import requests
import os
import urllib3
import time
import PyPDF2
import pdfplumber
from PyPDF2 import PdfReader
from typing import Optional

# Disable SSL warning (only if using verify=False)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_text_with_pdfplumber(file_path: str) -> Optional[str]:
    """
    Extract text from PDF using pdfplumber with better error handling
    """
    try:
        with pdfplumber.open(file_path) as pdf:
            text = []
            for page in pdf.pages:
                try:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text.append(extracted_text)
                except Exception as e:
                    print(f"Error extracting page: {str(e)}")
                    continue
            return "\n".join(text).strip()
    except Exception as e:
        print(f"pdfplumber extraction failed: {str(e)}")
        return None

def extract_text_with_pypdf2(file_path: str) -> Optional[str]:
    """
    Extract text from PDF using PyPDF2 with improved handling
    """
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            text = []
            for page in reader.pages:
                try:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text.append(extracted_text)
                except Exception as e:
                    print(f"Error extracting page: {str(e)}")
                    continue
            return "\n".join(text).strip()
    except Exception as e:
        print(f"PyPDF2 extraction failed: {str(e)}")
        return None

def extract_text_from_pdf(file_path, api_key):
    """
    Try multiple methods to extract text from PDF
    """
    # First try: Original API method
    try:
        # Submit endpoint
        submit_url = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper"
        # Retrieve endpoint
        retrieve_url = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper-retrieve"
        
        headers = {
            "unstract-key": api_key
        }
        
        with open(file_path, "rb") as pdf_file:
            file_data = pdf_file.read()
            
            try:
                # Submit the document
                response = requests.post(
                    submit_url, 
                    headers=headers, 
                    data=file_data,
                    timeout=30,
                    verify=False
                )
                
                if response.status_code == 202:
                    data = response.json()
                    whisper_hash = data.get("whisper_hash")
                    
                    # Save the whisper_hash to a file
                    with open("last_whisper_hash.txt", "w") as f:
                        f.write(f"{whisper_hash}\n")
                        f.write(f"File: {file_path}\n")
                        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    
                    print(f"Document submitted successfully. Processing...")
                    
                    # Poll for results
                    max_attempts = 60
                    attempt = 0
                    while attempt < max_attempts:
                        retrieve_response = requests.get(
                            retrieve_url,
                            headers=headers,
                            params={"whisper_hash": whisper_hash, "text_only": "true"},
                            verify=False
                        )
                        
                        if retrieve_response.status_code == 200:
                            return retrieve_response.text
                        
                        time.sleep(10)
                        attempt += 1
                    
                    print("API timeout, trying fallback methods...")
                    return None
                else:
                    print(f"API Error: {response.status_code}, trying fallback methods...")
                    return None
                    
            except requests.exceptions.RequestException as e:
                print(f"Network Error: {str(e)}, trying fallback methods...")
                return None
                
    except Exception as e:
        print(f"API method failed: {str(e)}, trying fallback methods...")
    
    # Second try: pdfplumber
    print("Attempting extraction with pdfplumber...")
    text = extract_text_with_pdfplumber(file_path)
    if text and len(text.strip()) > 0:
        return text
    
    # Third try: PyPDF2
    print("Attempting extraction with PyPDF2...")
    text = extract_text_with_pypdf2(file_path)
    if text and len(text.strip()) > 0:
        return text
    
    print("All extraction methods failed")
    return None

def check_extraction_status(whisper_hash, api_key):
    status_url = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper/status"
    headers = {
        "unstract-key": api_key
    }
    
    try:
        status_response = requests.get(
            f"{status_url}/{whisper_hash}",
            headers=headers,
            verify=False
        )
        
        print(f"Status Response Code: {status_response.status_code}")
        print(f"Full Response: {status_response.text}")
        
        if status_response.status_code == 200:
            result = status_response.json()
            if result.get("status") == "completed":
                return result.get("text", "")
            elif result.get("status") == "failed":
                print(f"Processing failed: {result.get('message')}")
                return None
            else:
                print(f"Status: {result.get('status', 'unknown')}")
                return None
    except Exception as e:
        print(f"Error checking status: {str(e)}")
        return None

def retrieve_text(whisper_hash, api_key):
    retrieve_url = "https://llmwhisperer-api.us-central.unstract.com/api/v2/whisper-retrieve"
    headers = {
        "unstract-key": api_key
    }
    
    try:
        retrieve_response = requests.get(
            retrieve_url,
            headers=headers,
            params={"whisper_hash": whisper_hash, "text_only": "true"},
            verify=False
        )
        
        print(f"Retrieve Response Code: {retrieve_response.status_code}")
        print(f"Full Response: {retrieve_response.text}")
        
        if retrieve_response.status_code == 200:
            return retrieve_response.text
        elif retrieve_response.status_code == 400:
            error_data = retrieve_response.json()
            error_message = error_data.get("message", "")
            print(f"Error: {error_message}")
        else:
            print(f"Unexpected status code: {retrieve_response.status_code}")
            print(f"Response: {retrieve_response.text}")
        
        return None
            
    except Exception as e:
        print(f"Error retrieving text: {str(e)}")
        return None

def process_pdfs(pdf_directory: str = "pdfs"):
    """
    Process all PDFs in the directory and save their texts
    """
    # Create output directory if it doesn't exist
    os.makedirs("extracted_texts", exist_ok=True)
    
    # Track results
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # Process each PDF file
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            output_path = os.path.join(
                "extracted_texts", 
                f"{os.path.splitext(filename)[0]}.txt"
            )
            
            # Skip if already processed
            if os.path.exists(output_path):
                print(f"Skipping {filename} - already processed")
                results['skipped'].append(filename)
                continue
                
            print(f"\nProcessing {filename}...")
            
            # Try pdfplumber first
            extracted_text = extract_text_with_pdfplumber(pdf_path)
            
            # If pdfplumber fails, try PyPDF2
            if not extracted_text or len(extracted_text.strip()) < 100:
                print("Trying PyPDF2...")
                extracted_text = extract_text_with_pypdf2(pdf_path)
            
            if extracted_text and len(extracted_text.strip()) > 100:
                # Save extracted text
                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(extracted_text)
                    print(f"✅ Saved text for {filename}")
                    results['success'].append(filename)
                except UnicodeEncodeError:
                    # Try saving with a different encoding if UTF-8 fails
                    with open(output_path, 'w', encoding='utf-16') as f:
                        f.write(extracted_text)
                    print(f"✅ Saved text for {filename} (UTF-16)")
                    results['success'].append(filename)
            else:
                print(f"❌ Failed to process {filename}")
                results['failed'].append(filename)
    
    # Print summary
    print("\n=== Processing Summary ===")
    print(f"Successfully processed: {len(results['success'])} files")
    print(f"Failed to process: {len(results['failed'])} files")
    print(f"Skipped (already processed): {len(results['skipped'])} files")
    
    if results['failed']:
        print("\nFailed files:")
        for file in results['failed']:
            print(f"- {file}")

# Example usage
if __name__ == "__main__":
    PDF_DIRECTORY = "D:/PycharmProjects/cpa files/pdfs"  # Replace with your actual path
    process_pdfs(PDF_DIRECTORY)
    print("\nAll PDFs processed! You can now run your Streamlit app.")




