import gradio as gr
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter

def process_input(questions):
    model_local = ChatOllama(model='mistral')

    # Split data into chunks
    urls = [
        "https://apnews.com/hub/donald-trump",
        "https://en.wikipedia.org/wiki/Donald_Trump",
        "https://www.britannica.com/biography/Donald-Trump",
    ]
    docs = []
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124"
    for url in urls:
        try:
            # Use XML parser for AP News, lxml for others
            parser_kwargs = {"features": "xml"} if "apnews" in url else {"features": "lxml"}
            loader = WebBaseLoader(url, bs_kwargs=parser_kwargs, requests_kwargs={"headers": {"User-Agent": user_agent}})
            loaded_docs = loader.load()
            if loaded_docs:
                docs.append(loaded_docs)
                print(f"Loaded {len(loaded_docs)} documents from {url}")
            else:
                print(f"No documents loaded from {url}")
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    if not docs:
        return "Error: No documents were loaded from any URLs."

    docs_list = [item for sublist in docs for item in sublist]
    print(f"Total documents: {len(docs_list)}")

    # Filter out empty documents
    docs_list = [doc for doc in docs_list if doc.page_content.strip()]
    if not docs_list:
        return "Error: All documents are empty after filtering."
    print(f"Non-empty documents: {len(docs_list)}")

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    docs_splits = text_splitter.split_documents(docs_list)
    print(f"Total document chunks: {len(docs_splits)}")

    if not docs_splits:
        return "Error: No document chunks created."

    # Convert documents to embeddings and store them
    try:
        vector_store = Chroma.from_documents(
            documents=docs_splits,
            collection_name="rag-chroma",
            embedding=OllamaEmbeddings(model='nomic-embed-text'),
            persist_directory="./chroma_db"
        )
        retriever = vector_store.as_retriever()
    except Exception as e:
        return f"Error creating vector store: {e}"

    # Using RAG
    print("****\nResponse\n")
    rag_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    rag_prompt = ChatPromptTemplate.from_template(rag_template)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | rag_prompt
        | model_local
        | StrOutputParser()
    )
    return rag_chain.invoke(questions)

# Define Gradio Interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(label="Enter the Question Here")],
    outputs="text",  # Fixed from "texts" to "text"
    title="All about Donald Trump",
    description="Ask questions about Donald Trump based on web sources."
)
iface.launch()