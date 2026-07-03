# Medical RAG Assistant

Retrieval-Augmented Generation system for medical Q&A, built on the Merck Manual 
of Diagnosis and Therapy. Upload any medical reference PDF and ask clinical questions
the system retrieves relevant passages and generates grounded answers.

## Live Demo

[▶ Open Interactive App]https://medical-rag-assistant-6i7b25wv6tam7x9xhulbsf.streamlit.app/

## Background

This was my NLP/RAG coursework project, originally built in Google Colab using 
**Meta Llama 3.1 8B Instruct** (Q5_K_M quantized GGUF) on a Tesla T4 GPU, with 
ChromaDB as the vector store and LangChain as the orchestration framework.

The deployed demo replaces Llama with **Claude Haiku** as the generation backend 
for accessibility, the full RAG pipeline (chunking, embedding, retrieval) is 
unchanged. This project directly preceded and informed the design of the 
[research_assistant](https://github.com/xmashaxxx/research_assistant) agentic pipeline.

## Technical Stack

| Component | Coursework | This Demo |
|-----------|-----------|-----------|
| Document loading | PyMuPDF | PyMuPDF |
| Chunking | RecursiveCharacterTextSplitter | RecursiveCharacterTextSplitter |
| Embeddings | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 |
| Vector store | ChromaDB | ChromaDB |
| Generation | Llama 3.1 8B (quantized) | Claude Haiku |
| Infrastructure | Google Colab + Tesla T4 | Streamlit Cloud |

## What Was Evaluated

5 parameter combinations tested across 5 clinical questions (sepsis protocol, 
pulmonary embolism, rheumatoid arthritis, endocrine disorders, hypertension):

| Combination | Settings | Key Finding |
|-------------|----------|-------------|
| 1 (baseline) | temperature=0 | Deterministic, structured, shortest |
| 2 | temperature=0.7 | Longer, more detailed |
| 3 | temperature=0.5, top_p=0.85, top_k=40 | Most focused, least repetitive |
| 4 | temperature=0, repeat_penalty=1.3 | Baseline quality, less redundancy |
| 5 | temperature=0.9, seed=99 | Most varied, added novel sections |

**Key finding:** temperature=0 with parameter tuning produced the most clinically 
reliable responses. Higher temperatures surfaced more detail but risked adding 
information beyond the retrieved context.

## Running Locally

```bash
git clone https://github.com/xmashaxxx/medical-rag-assistant.git
cd medical-rag-assistant
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here
streamlit run app.py
```

Upload any medical reference PDF to use the system.

## Dataset

See [`data/README.md`](data/README.md). The Merck Manual is proprietary and 
not included, but the app accepts any medical PDF.

## Connection to research_assistant

This coursework directly informed the [research_assistant](https://github.com/xmashaxxx/research_assistant) 
agentic pipeline. Key lessons carried forward: RAG reduces hallucination vs. 
closed-book LLM, chunking strategy significantly affects retrieval quality, 
and separating retrieval from generation allows independent optimization of each stage.

## Skills Demonstrated

`Python` `LangChain` `ChromaDB` `RAG` `sentence-transformers` `PyMuPDF` 
`Anthropic API` `Claude Haiku` `Streamlit` `NLP` `Parameter Evaluation`
