# PaperTrail AI

**An evidence-grounded research paper intelligence engine.**

PaperTrail AI aims to help researchers analyze scientific papers, retrieve relevant evidence, compare methodologies, and generate answers supported by source citations.

## Planned Features

- PDF ingestion with page-level metadata
- Text cleaning and overlapping chunking
- Semantic retrieval using embeddings and FAISS
- Keyword retrieval using BM25
- Hybrid retrieval and cross-encoder reranking
- Evidence-grounded question answering
- Structured extraction of research methods, datasets, metrics, findings, and limitations
- Cross-paper comparison
- Retrieval and citation evaluation

## Technology Stack

Python · PyMuPDF · Sentence Transformers · FAISS · BM25 · LLM APIs · Pydantic · Streamlit

## Project Status

**Week 1 — In progress**

Current milestone: PDF text extraction with page-level metadata.

## Planned Evaluation

- Recall@K
- nDCG@K
- Mean Reciprocal Rank (MRR)
- Citation validity and support
- Structured extraction accuracy
- Latency and API cost

Evaluation results will be added after experiments have been implemented and run.

## Getting Started

The installation instructions will be finalized as the project dependencies and pipeline are implemented.

## Roadmap

- [ ] PDF ingestion
- [ ] Text cleaning and chunking
- [ ] Dense retrieval with FAISS
- [ ] BM25 keyword retrieval
- [ ] Hybrid retrieval and reranking
- [ ] Evidence-grounded question answering
- [ ] Research paper comparison
- [ ] Evaluation and reproducibility
- [ ] Streamlit interface

## Disclaimer

This is an experimental research assistant. Generated answers and extracted information require verification against the original papers.