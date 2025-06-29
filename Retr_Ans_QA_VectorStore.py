from QA_VectorStore import *
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI

def relevant_QA_database(query) :

    vector_store = Create_and_load_QA_Data()

    # Step 5: Query the vector store
    def query_vector_store(query):
        """Query the vector store with the specified question."""
        # Create a retriever for querying the vector store
        retriever = vector_store.as_retriever(
            search_type="similarity"
        )

        # Retrieve relevant documents based on the query
        relevant_docs = retriever.invoke(query)

        # Display the relevant results with metadata
        # print("\n--- Relevant Documents ---")
        # for i, doc in enumerate(relevant_docs, 1):
        #     print(f"Document {i}:\n{doc.page_content}\n")
        #     if doc.metadata:
        #         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
        return relevant_docs

    return query_vector_store(query)

# print(relevant_QA_database("Severe headache with nausea"))








