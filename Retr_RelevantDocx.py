from Retr_Ans_QA_VectorStore import*
from Retr_Ans_Web_VectorStore import*

query = "I have a one sided headache that feels like its beating, I vomited an hour ago and am continuously feeling nauseous"

docsfromQA = relevant_QA_database(query)
docsfromWeb = relevant_Web_database(query)

print("\n--- Relevant Documents from QA database ---")
for i, doc in enumerate(docsfromQA, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

print("\n--- Relevant Documents from Web database ---")
for i, doc in enumerate(docsfromWeb, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    if doc.metadata:
        print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

