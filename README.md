# Transcript Intelligence

A local app for exploring three robotic-surgery expert interviews. The files represent the **France, Germany, and United Kingdom markets**; all three transcripts are written in English. Select a market to scope a question, or use the interview guide and comparison views.

## Run locally

You need Python, Node.js/npm, and an OpenRouter API key. The default configuration uses OpenRouter for BAAI BGE-M3 embeddings and DeepSeek structured answers.

1. Configure the backend:

   ```sh
   cd backend
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env
   ```

   Add your `OPENROUTER_API_KEY` to `backend/.env`. If that file already exists, keep it and add the setting there instead of replacing it.

2. Start the backend and index the transcripts:

   ```sh
   uvicorn src.api:app --reload --port 8000
   ```

   In the app, choose **Index transcripts**. This reads all `.txt` files from `backend/data/raw` and persists the local index. You can also index from another terminal with:

   ```sh
   curl -X POST http://localhost:8000/api/ingest
   ```

3. In a second terminal, start the frontend:

   ```sh
   cd frontend
   npm install
   cp .env.local.example .env.local
   npm run dev
   ```

   Open <http://localhost:3000>.

## What the app does

- **Ask interviews:** Ask in English and choose All markets, France, Germany, or United Kingdom. Country scope filters both semantic and keyword retrieval. The question language does not choose a market.
- **Interview guide:** Generate the six guide answers separately for every indexed expert. Each answer is validated against that expert's transcript evidence and includes source citations.
- **Compare all three:** Synthesize shared themes and differences using the complete evidence from all three markets.
- **Citations:** Each claim points to a short verbatim answer excerpt, the source file, and the answer timestamp. Claims are rendered from the validated claim list so citations stay attached to the claim they support.

## Architecture and grounding

FastAPI parses timestamped interviewer/expert turns, then builds a persistent Chroma semantic index and BM25 keyword index. Questions are planned, retrieved within any selected country scope, fused, reranked, and sent to the answer model with source evidence IDs. The API constructs citations from indexed metadata rather than asking the model to invent citation details.

The validator rejects unknown evidence IDs, unsupported numbers, absent quoted text, and claims with insufficient source-word overlap. It assembles the displayed answer only from validated claims and abstains when evidence is missing. This is a useful deterministic guard, not a complete semantic proof that every paraphrase is entailed; review the linked source excerpts for high-confidence use.

For a larger archive, retain the same metadata and source-ID contracts, batch embedding/index writes, and use asynchronous jobs plus cached guide/comparison results. The current guide and comparison flows are designed for the supplied three short transcripts, not a 30-transcript batch workload.
