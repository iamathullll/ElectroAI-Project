import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

# Load and prepare data once
urls = [
    "https://apnews.com/hub/donald-trump",
    "https://en.wikipedia.org/wiki/Donald_Trump",
    "https://www.britannica.com/biography/Donald-Trump",
    "https://www.facebook.com/DonaldTrump/",
    "https://x.com/realdonaldtrump"
]

print("Loading documents...")
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
docs_splits = text_splitter.split_documents(docs_list)

# Create vector store
embedding_model = OllamaEmbeddings(model='nomic-embed-text')
vector_store = Chroma.from_documents(
    documents=docs_splits,
    collection_name="rag-chroma",
    embedding=embedding_model
)
# convert vector store into a retriever
retriever = vector_store.as_retriever()

# Prompt template
rag_template = """Answer the question based only on the following context:
{context}
Question: {question}
"""
rag_prompt = ChatPromptTemplate.from_template(rag_template)

# RAG chain
model_local = ChatOllama(model='mistral')
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | model_local
    | StrOutputParser()
)

# Gradio interface
def process_input(question):
    return rag_chain.invoke(question)

iface = gr.Interface(
    fn=process_input,
    inputs=gr.Textbox(label="Enter the Question Here"),
    outputs=gr.Textbox(label="Answer"),
    title="All About Donald Trump"
)

iface.launch()
# /usr/local/bin/python3 /Users/athulkrishnavv/Desktop/ollama/TrumpBot.py
