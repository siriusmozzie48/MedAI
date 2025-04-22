from QA_VectorStore import *
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Define the persistent directory

def relevant_QA_database(query) :

    vector_store = Create_and_load_QA_Data()

    # # Load the vector store with the embeddings
    # embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

    # load_dotenv()

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

    # Define the user's question

    # Query the vector store with the user's question
    #query_vector_store(query)

    return query_vector_store(query)
# print(relevant_docs)

# Combine the query and the relevant document contents
# combined_input = (
#     "Use the documents provided under 'Relevant Documents' containing a medical question and answer pair accompanied with the question type to generate a response that tries to figure out the medical condition of the user as exhibited by the symptoms described by the user : "
#     + query
#     + "\n\nRelevant Documents:\n"
#     + "\n\n".join([doc.page_content for doc in relevant_docs])
#     + "\n\nPlease generate a response using only the provided 'Relevant Documents'. If more information is required from the user that satisfies the medical condition as in the 'Relvant Documetns', prompt the user for that particular information. If there is no relevance in the documents and you are unable to find the answer from the documents, respond with 'I'm not sure'."
# )

# Create a ChatOpenAI model
# client = ChatNVIDIA(
#     model="deepseek-ai/deepseek-r1",
#     api_key="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",  # Replace with your NVIDIA API key
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=4096
# )

# client = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.7,
#     top_p=0.8,
#     max_tokens=4096
# )

# # Define the messages for the model
# messages = [
#     SystemMessage(content="You are a medical assistant that uses given 'Relevant Documents' and only them to determine the medical condition suffered by the user,  If more information is required from the user that satisfies the medical condition as in the 'Relvant Documetns', prompt the user for that particular information. If there is no relevance in the documents and you are unable to find the answer from the documents, respond with 'I'm not sure'  after every response, you give a disclaimer 'This is an AI generated response, please consult a medical expert before taking any actions'"),
#     HumanMessage(content=combined_input),
# ]

# # Invoke the model with the combined input
# result = client.invoke(messages)

# # Display the full result and content only
# print("\n--- Generated Response ---")
# # print("Full result:")
# # print(result)
# print("Content only:")
# print(result.content)

    








