# Dataset

**Source:** The Merck Manual of Diagnosis and Therapy
**Format:** PDF, 4000+ pages, 23 sections
**Status:** Proprietary — not included in this repository

The deployed demo accepts any medical PDF as input.

## Pipeline
1. Load and chunk the PDF
2. Embed chunks using all-MiniLM-L6-v2
3. Store in ChromaDB
4. Retrieve top-k chunks per query
5. Generate answer with Claude Haiku
