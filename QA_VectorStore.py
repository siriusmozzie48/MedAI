import os
import pandas as pd
import re
import time
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
FOLDER_PATH = "C:\\Users\\abdul\\Downloads\\Crawl4ai\\Project\\Databases"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
BATCH_SIZE = 100  # Number of documents processed per batch
SAVE_PATH = os.path.join(FOLDER_PATH, "QA_db")  # FAISS storage file # FAISS storage file

# --- Initialize Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# --- Load CSV Data ---
def load_csv_data(folder_path):
    csv_files = [file for file in os.listdir(folder_path) if file.endswith('.csv')]
    dataframes = [pd.read_csv(os.path.join(folder_path, file), usecols=['Question', 'Answer', 'qtype']) for file in csv_files]
    df = pd.concat(dataframes, ignore_index=True)
    return df

# --- Clean Text (Remove New Lines and Extra Spaces) ---
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces/newlines with a single space
    return text.strip()

# --- Preprocess Data (Format for Embedding) ---
def preprocess_data(df):
    documents = []
    for index, row in df.iterrows():
        question = clean_text(row["Question"])
        answer = clean_text(row["Answer"])
        qtype = clean_text(row["qtype"])
        text = f"Question: {question}, Answer: {answer}, Question-Type: {qtype}"
        documents.append(text)

        # Log progress every 1000 entries
        if index % 1000 == 0:
            print(f"✅ Processed {index}/{len(df)} documents")

    return documents

# --- Embed and Store Data in FAISS ---
def create_faiss_index(documents):
    total_batches = (len(documents) + BATCH_SIZE - 1) // BATCH_SIZE
    vector_store = None

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        print(f"🚀 Processing batch {i // BATCH_SIZE + 1}/{total_batches} ({len(batch)} documents)...")

        start_time = time.time()
        if vector_store is None:
            vector_store = FAISS.from_texts(batch, embedding_model)
        else:
            new_store = FAISS.from_texts(batch, embedding_model)
            vector_store.merge_from(new_store)

        print(f"✅ Batch {i // BATCH_SIZE + 1} completed in {time.time() - start_time:.2f} sec")

    # Save FAISS index
    vector_store.save_local(SAVE_PATH)
    print(f"✅ FAISS index saved to {SAVE_PATH}")


def load_faiss_vectorstore():
    """
    Loads an existing FAISS vector store.
    
    Returns:
        FAISS vector store instance (if exists), otherwise None.
    """
    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    try:
        # Load FAISS index
        vector_store = FAISS.load_local(SAVE_PATH, embedding_model, allow_dangerous_deserialization=True)
        print(f"✅ Vector store loaded successfully with {len(vector_store.index_to_docstore_id)} vectors.")
        return vector_store
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


def Create_and_load_QA_Data():
        print("🚀 Loading CSV data...")
        df = load_csv_data(FOLDER_PATH)
        print(f"📊 Loaded {len(df)} entries")

        print("🔄 Preprocessing data...")
        documents = preprocess_data(df)
        print(f"✅ Finished processing {len(documents)} documents")
        if os.path.exists(SAVE_PATH):
            print("Vector darabase already exists")
        else :
            print("⚡ Embedding documents and storing in FAISS...")
            create_faiss_index(documents)
            print("🎉 Done!")
        return load_faiss_vectorstore()


    
