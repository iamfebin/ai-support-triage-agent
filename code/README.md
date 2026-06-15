# AI Support Triage Agent

This is a highly resilient, cost-effective AI agent designed to triage support tickets for HackerRank, Claude, and Visa. It processes a dataset of tickets, intelligently retrieves relevant support documentation, and utilizes an LLM to generate precise, structured JSON responses.

## Architecture Overview

Our agent employs three core architectural features to ensure stability and accuracy:

1. **Local Semantic Search (Zero-Cost Retrieval):** 
   Instead of using external API-based embeddings which consume tokens, the agent uses the local `sentence-transformers` library (`all-MiniLM-L6-v2`) and `numpy` to generate embeddings and calculate cosine similarity. This ensures highly accurate, free retrieval of support documentation.
   
2. **High-Availability Model Fallback:** 
   The agent utilizes Gemini's Structured Outputs (via Pydantic) to guarantee valid JSON. It defaults to `gemini-3.1-flash-lite-preview` for speed and efficiency. However, if a `503` (Service Unavailable) or `429` (Rate Limit) error is detected during its exponential backoff retry loop, it automatically downgrades to `gemini-2.5-flash` to ensure maximum availability.

3. **Idempotent Execution & Checkpointing:** 
   To prevent data loss, the agent writes each ticket directly to `output.csv` immediately after processing using a `csv.DictWriter` in append mode. If the script is interrupted, running it again will automatically read `output.csv`, skip successfully processed tickets, and ONLY retry tickets that failed or encountered rate limits.

---

## Setup Instructions

**1. Create a Virtual Environment (Recommended)**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

**2. Install Dependencies**
Install all required packages from the pinned requirements file:
```bash
pip install -r requirements.txt
```

**3. Configure Environment Variables**
Create a `.env` file in the root directory (alongside `AGENTS.md`) and add your Gemini API Key:
```env
GEMINI_API_KEY=your_api_key_here
```

---

## Running the Agent

To start the triage process on the `support_tickets.csv` dataset, simply run the main script from the root directory:

```bash
python code/main.py
```

The script will:
1. Load the support corpus and generate local embeddings.
2. Read the ticket dataset.
3. Skip any tickets that have already been successfully processed (if resuming).
4. Process tickets one by one, checkpointing them in real-time to `support_tickets/output.csv`.
