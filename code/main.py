import os
import time
import json
import pandas as pd
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
# Switching to the updated 2026 library to resolve the warning
from google import genai 
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load API Key from .env
load_dotenv()
# Initialize the updated client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Initializing local semantic search model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

class TriageResult(BaseModel):
    status: str = Field(description="Must be exactly 'replied' or 'escalated'")
    product_area: str
    response: str = Field(description="Use ONLY the provided corpus. If info is missing or sensitive, set status to 'escalated'.")
    justification: str
    request_type: str = Field(description="Must be exactly 'product_issue', 'feature_request', 'bug', or 'invalid'")

def load_corpus():
    """
    Reads text files from domain sub-folders in data/ and returns a dictionary.
    Format: corpus[domain][filename] = content
    """
    corpus = {'hackerrank': {}, 'claude': {}, 'visa': {}}
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data') 
    
    if not os.path.exists(base_path):
        print(f"Error: Data directory not found at {base_path}")
        return corpus

    for domain in corpus.keys():
        domain_path = os.path.join(base_path, domain)
        if os.path.exists(domain_path):
            for root, dirs, files in os.walk(domain_path):
                for file in files:
                    if file.endswith((".txt", ".md")):
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            text = f.read()
                            embedding = embedder.encode(text)
                            corpus[domain][file] = {
                                "text": text,
                                "embedding": embedding
                            }
    return corpus

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search_corpus(issue, subject, domain_corpus, top_k=3):
    """
    Local semantic search using sentence-transformers to pick top_k relevant files.
    """
    text = f"{str(issue)} {str(subject)}"
    query_embedding = embedder.encode(text)
    
    scores = {}
    for filename, data in domain_corpus.items():
        doc_embedding = data["embedding"]
        score = cosine_similarity(query_embedding, doc_embedding)
        scores[filename] = score
        
    # Sort files by score descending
    sorted_files = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    # Pick top_k files
    top_files = sorted_files[:top_k]
    
    # Construct final sub-corpus
    parts = []
    for fname, _ in top_files:
        parts.append(f"Source: {fname}\n{domain_corpus[fname]['text']}\n")
        
    return "\n".join(parts)

def triage_ticket(ticket_row, corpus):
    prompt = f"""
    You are a professional support agent triaging tickets. Use ONLY the provided support corpus to answer.
    
    SUPPORT CORPUS:
    {corpus}

    TICKET DETAILS:
    Issue: {ticket_row.get('issue', '')}
    Subject: {ticket_row.get('subject', '')}
    Company: {ticket_row.get('company', '')}

    CONSTRAINTS:
    1. If Company is 'None' or missing, infer it from the issue text (HackerRank, Claude, or Visa).
    2. DO NOT hallucinate. 
    
    EXAMPLES:
    Example 1 (Replied):
    {{
        "status": "replied",
        "product_area": "Billing",
        "response": "To update your billing, go to settings and click billing.",
        "justification": "The corpus specifically provides these exact billing instructions.",
        "request_type": "product_issue"
    }}
    
    Example 2 (Escalated):
    {{
        "status": "escalated",
        "product_area": "Security",
        "response": "Error processing ticket.",
        "justification": "The user asks for sensitive internal fraud rules which are not in the corpus.",
        "request_type": "invalid"
    }}
    """
    
    current_model = "gemini-3.1-flash-lite-preview"
    wait_times = [2, 4, 8, 16, 65]
    max_retries = len(wait_times) + 1
    for attempt in range(max_retries):
        try:
            # Using the modern 2026 SDK syntax with Structured Outputs
            response = client.models.generate_content(
                model=current_model,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TriageResult,
                    'temperature': 0.0
                }
            )
            print(f"RAW LLM OUTPUT: {response.text}")
            
            if not response.text or not response.text.strip():
                raise ValueError("Empty response text received from the model.")
                
            return json.loads(response.text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str or "503" in error_str or "Service Unavailable" in error_str:
                current_model = "gemini-2.5-flash" # Fallback routing
                if attempt < len(wait_times):
                    wait_time = wait_times[attempt]
                    print(f"API Error ({error_str[:30]}...). Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return {
                        "status": "escalated",
                        "product_area": "error",
                        "response": "Error processing ticket.",
                        "justification": "Rate limit exceeded after retries",
                        "request_type": "invalid"
                    }
            else:
                return {
                    "status": "escalated",
                    "product_area": "error",
                    "response": "Error processing ticket.",
                    "justification": str(e),
                    "request_type": "invalid"
                }
                
    return {
        "status": "escalated",
        "product_area": "error",
        "response": "Error processing ticket.",
        "justification": "Rate limit exceeded after retries",
        "request_type": "invalid"
    }

if __name__ == "__main__":
    # Corrected paths relative to the project structure
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SAMPLE_PATH = os.path.join(BASE_DIR, 'support_tickets', 'sample_support_tickets.csv')
    FINAL_PATH = os.path.join(BASE_DIR, 'support_tickets', 'support_tickets.csv')
    OUTPUT_PATH = os.path.join(BASE_DIR, 'support_tickets', 'output.csv')

    print("Loading corpus...")
    corpus = load_corpus()
    
    # Load final evaluation data
    print(f"Loading tickets from {FINAL_PATH}...")
    tickets = pd.read_csv(FINAL_PATH)

    # Normalize columns to lowercase as per schema
    tickets.columns = [c.lower() for c in tickets.columns]
    
    # Ensure ONLY the required columns are in the final output
    required_cols = [
        'status', 'product_area', 'response', 'justification', 
        'request_type', 'issue', 'subject', 'company'
    ]
    
    output_exists = os.path.exists(OUTPUT_PATH)
    
    # --- IDEMPOTENT EXECUTION LOGIC ---
    processed_issues = set()
    if output_exists:
        try:
            df_existing = pd.read_csv(OUTPUT_PATH)
            for _, r in df_existing.iterrows():
                justification = str(r.get('justification', ''))
                response_text = str(r.get('response', ''))
                # Do NOT add to successful set if there was an error
                if "Rate limit exceeded" not in justification and "Error processing ticket." not in response_text and "Error processing ticket." not in justification:
                    processed_issues.add(str(r.get('issue', '')))
        except pd.errors.EmptyDataError:
            pass # File is empty
    # -----------------------------------
    
    print(f"Processing {len(tickets)} tickets. ({len(processed_issues)} already completed)")
    
    with open(OUTPUT_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=required_cols)
        if not output_exists or os.path.getsize(OUTPUT_PATH) == 0:
            writer.writeheader()
            
        for index, row in tickets.iterrows():
            issue_text = str(row.get('issue', ''))
            if issue_text in processed_issues:
                print(f"Skipping {index + 1}/{len(tickets)} (Already Processed): {row.get('subject', 'No Subject')}")
                continue
                
            print(f"Triaging {index + 1}/{len(tickets)}: {row.get('subject', 'No Subject')}")
            
            company = str(row.get('company', '')).lower()
            
            # Smart Routing: Combine domains if unknown
            if 'hackerrank' in company:
                domain_corpus = corpus.get('hackerrank', {})
            elif 'claude' in company:
                domain_corpus = corpus.get('claude', {})
            elif 'visa' in company:
                domain_corpus = corpus.get('visa', {})
            else:
                # Unknown company, combine all corpuses
                domain_corpus = {}
                for d_corp in corpus.values():
                    domain_corpus.update(d_corp)
            
            sub_corpus = search_corpus(row.get('issue', ''), row.get('subject', ''), domain_corpus)
            result = triage_ticket(row, sub_corpus)
            
            # Combine original data with agent output
            combined = {**row.to_dict(), **result}
            
            # Filter dict to only required columns
            filtered_combined = {k: v for k, v in combined.items() if k in required_cols}
            
            # Checkpoint directly to CSV
            writer.writerow(filtered_combined)
            f.flush() # Ensure it's written to disk immediately
            
            time.sleep(15) # Safe delay to avoid rate limits
        
    print(f"Done! Results saved to {OUTPUT_PATH}")

