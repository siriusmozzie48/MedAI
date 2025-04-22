from two import *
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from condition import *
from WebSearch import *

FOLDER_PATH = "C:\\Users\\abdul\\Downloads\\Crawl4ai\\Project\\Databases"
SAVE_PATH = os.path.join(FOLDER_PATH, "Web_db")  # FAISS storage file # FAISS storage file
EMBEDDINGMODEL = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")
# Define the persistent directory

load_dotenv()

client = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    top_p=0.8,
    max_tokens=4096
)
    
def crawl_web(query):
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    link, title, snippet = WebSearch(query)
    condition = detect_conditions(link, title, snippet)
    print("---Crawling the website----\n")
    content = asyncio.run(information_from_url(link, condition))
    # print(content)
    documents = [Document(page_content=chunk) for chunk in content]
    print("Finished Crawling...adding metadata\n")
    for doc in documents:
    # Add metadata to each document indicating its source
        doc.metadata = {"source": link}

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(documents)}")
    print(f"Sample chunk:\n{documents[0].page_content}\n")

    #Create embeddings for the document chunks

    #Create and persist the vector store with the embeddings
    print(f"\n--- Creating vector store in {SAVE_PATH} ---")
    vector_store = FAISS.from_documents(
        documents, EMBEDDINGMODEL
    )
    vector_store.save_local(SAVE_PATH)
    print(f"--- Finished creating vector store in {SAVE_PATH} ---")

# Check if the Chroma vector store already exists

def Create_Web_Vector_Store(query):

    if not os.path.exists(SAVE_PATH):
        crawl_web(query)
    else:
        print(
            f"Vector store {SAVE_PATH} already exists. No need to initialize.")
    vector_store = FAISS.load_local(SAVE_PATH, EMBEDDINGMODEL, allow_dangerous_deserialization=True)
    return vector_store


# Load the vector store with the embeddings


    








