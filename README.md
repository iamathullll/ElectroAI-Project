# AI/ML Internship Assessment – ElectoAI Analytics Magnifier Pvt Ltd

## Project Title: Celebrity Insight & Sentiment Analysis (Donald Trump)

## Task Overview

This project is submitted in response to the AI/ML Internship Assessment by ElectoAI Analytics Magnifier Pvt Ltd. The objective is to build an AI-driven system capable of understanding a chosen celebrity by analyzing online data from the past 30 days. The system is designed to extract sentiment, context, patterns, and answer user queries based on the data.

---

## Objectives

- Select a celebrity (Donald Trump) and collect recent online data.
- Preprocess and clean the text data.
- Perform sentiment analysis.
- Build a Retrieval-Augmented Generation (RAG) pipeline for Q&A.
- Visualize insights using sentiment and frequency charts.

---

## Selected Celebrity: Donald Trump

Reason: Due to frequent news appearances and political engagement, Donald Trump has had substantial coverage in media sources, making him a suitable candidate for data-driven AI analysis.

---

## Data Sources

The following websites were used to gather textual data:

- https://apnews.com/hub/donald-trump
- https://en.wikipedia.org/wiki/Donald_Trump
- https://www.britannica.com/biography/Donald-Trump
- https://www.facebook.com/DonaldTrump/

Data was fetched and parsed using `WebBaseLoader` from the `langchain_community` library.

---

## Data Preprocessing

- Merged and flattened document data from all sources.
- Used `CharacterTextSplitter` to chunk documents into manageable segments.
- Removed irrelevant text, noise, and standardized text format.

---

## Model and Pipeline

- Embeddings: `nomic-embed-text` via `OllamaEmbeddings`
- Vector Store: `Chroma` for storing text chunks and enabling retrieval.
- Language Model: `ChatOllama` (using the `mistral` model)
- RAG Chain: Combined retriever, prompt, and model into a LangChain-based inference chain.
- Interface: Interactive Q&A interface using Gradio.

---

## Sentiment Analysis & Visualization

Used Hugging Face’s `pipeline("text-classification")` for sentiment scoring, categorized into:

- Positive
- Negative
- Neutral

Visualizations include:

- Sentiment distribution (bar chart)
- Sentiment trend over time (line graph)
- Keyword frequency (word cloud)

---

## How to Run

1. Install dependencies:
# Core dependencies
pandas
scikit-learn
numpy

# Hugging Face Transformers for sentiment analysis
transformers
torch  # required for model execution

# Gradio for the web interface
gradio

# LangChain components
langchain
langchain-community
langchain-core

# Embedding and vector store support
chromadb
ollama  # required to run local models via ChatOllama and OllamaEmbeddings

# Visualization libraries
matplotlib
seaborn
wordcloud

# Web scraping/loader support
beautifulsoup4
html5lib
requests

