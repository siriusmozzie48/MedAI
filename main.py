from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from firecrawlSearch import *
from Web_VectorStore import *
from Retr_Ans_QA_VectorStore import relevant_QA_database


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_output_tokens=4096 
)

def analyze_query(query: str) -> str:
    """
    Analyzes the user's query to determine if it's medical and extracts symptoms.
    """
    system_prompt = """
    You are a helpful medical assistant. Your task is to analyze the user's query.
    1.  Determine if the query is medically related.
    2.  If it is medical and descriptive, identify and list the main symptoms described.
    3.  If the query is not descriptive and contains insufficient information (e.g., "I feel sick"), Classify the query as "Vague Medical Query" and exemplify the descriptions.
    4.  If the query has nothing to do with a medical condition, does not describe symptoms or is irrelevant for example "I love rainbows", simply classify it as 'non-medical'.
    
    Respond in one of the following formats:
    - If medical and descriptive: "Medical Query. Symptoms: [list of symptoms]"
    - If medical but vague and lacks description: "Vague Medical Query,For Example:(sample description)."
    - If not medical or irrelevant: "Non-Medical Query"
    """

    analysis_result = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ])
    
    return analysis_result.content

def reformulate_query_with_history(query: str, chat_history: list) -> str:
    """
    Reformulates a follow-up query into a standalone question using chat history.
    This version uses a more robust prompt to generate a query ideal for semantic search.
    """
    if not chat_history:
        return query
    
    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the conversation above, create a single, natural language, self-contained search query to find relevant medical documents from a vector database.
         Combine the user's latest message with relevant context from the chat history.The query should be precise, to the point and rich with symptoms and medical keywords so that it can effectively retrieve relevant information using embeddings.
         Do not answer the question or include anything else. Only output the reformulated search query."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    chain = history_aware_prompt | llm
    result = chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    
    return result.content

def reformulate_query_for_Search(query: str, chat_history: list) -> str:
    """
    Reformulates a follow-up query into a standalone question using chat history.
    This version uses a more robust prompt to generate a query ideal for search engines.
    """
    if not chat_history:
        return query

    history_aware_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given the conversation above, create an optimized natural language search query for a search engine to find relevant links from medical websites to scrape information relevant to the query.
Combine the user's latest message with relevant context from the chat history.
The query should be a descriptive question search or a statement rich with symptoms and medical keywords.
Do not answer the question or include anything else. Only output the reformulated search query."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    
    chain = history_aware_prompt | llm
    result = chain.invoke({
        "chat_history": chat_history,
        "input": query
    })
    
    return result.content

def generate_conversational_response(query: str, all_docs: list, chat_history: list) -> str:
    """
    Generates a helpful and safe response based on the retrieved documents and chat history.
    """
    final_context_string = "\n".join([doc.page_content for doc in all_docs])
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are an expert, compassionate, and knowledgeable medical information assistant.
    Your goal is to synthesize the provided `Retrieved Documents` and `chat_history` to give the user a clear, structured, and helpful response.

    **Your Task & Response Structure:**

    Based on the user's conversation and the retrieved documents, structure your response using the following markdown format. If a section is not relevant or you don't have enough information from the documents, omit that section.

    ### Potential Condition(s)
    Briefly summarize the potential medical condition(s) suggested by the documents that align with the user's symptoms.

    ### Key Information from Retrieved Documents
    - **Possible Causes**: Summarize potential causes mentioned in the documents.
    - **Common Symptoms**: List key symptoms described in the documents. Be sure to mention any that the user has also reported.
    - **Suggested Treatments or Remedies**: Outline possible treatments or home remedies suggested by the documents.

    ### Next Steps & Recommendations
    Based on the documents, provide actionable advice. This may include:
    - **When to See a Doctor**: List any "red flag" symptoms or conditions under which the documents recommend seeking professional medical help.
    - **Preventative Measures**: If mentioned in the documents, list any preventative measures.

    **Critical Safety Instructions:**
    1.  **Grounding**: Base your entire response ONLY on the information found in the `Retrieved Documents` and the `chat_history`. Do not add any information from your own knowledge.
    2.  **Disclaimer**: You MUST end your entire response, on a new line, with the following mandatory disclaimer. There are no exceptions.

    ***
    ***Disclaimer: This is an AI-generated response and not a substitute for professional medical advice. Please consult a doctor or other qualified healthcare provider for any medical concerns.***
    Here is the context from the retrieved documents:\n"{context}\n"
         """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt_template | llm
    
    response = chain.invoke({
        "chat_history": chat_history,
        "input": query,
        "context": final_context_string
    })
    
    return response.content

def needs_web_search(query: str, retrieved_docs: list, chat_history: list) -> bool:
    """
    Uses an LLM to decide if the currently retrieved documents are sufficient
    or if a web search is required. This version is compatible with the Gemini API.
    """

    print("Decding if a web search is needed based on retrieved documents...\n")

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    if not retrieved_docs:
        print("--- No local documents found. A web search is necessary. ---")
        return True
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical information assistant. Your task is to decide if the information retrieved from the local database is sufficient to help the user understand their symptoms, potential medical conditions, and possible treatments, or if an external web search is needed.

**Instructions:**
- Carefully review the "Retrieved Local Documents" below, the user's conversation history, and their latest input (which may be a question or a description of symptoms).
- If the local documents provide fully addresses the user's needs—including explanations of symptoms, possible conditions, and treatments—respond with "NO".
- If the documents are missing key details, are incomplete, or do not address the user's specific situation, respond with "YES".

**Important:** Respond with only the word "YES" or "NO".

**--- Retrieved Local Documents ---**
{context}
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])

    chain = prompt | llm
    response = chain.invoke({
        "chat_history": chat_history,
        "input": query,
        "context": context
    })

    decision = response.content.strip().upper()
    print(f"--- Web search decision: {decision} ---\n")
    return decision == "YES"



def main():
    print("Assistant : Initialized. How are you feeling today? Deescribe your symptoms or concerns, and I will do my best to assist you with medical information.")
    
    chat_history = []
    
    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye! Stay healthy.")
            break

        print("\nAnalyzing your query...")
        analysis = analyze_query(query)
        # print(f"Analysis:\n{analysis}\n")
        
        if "Non-Medical Query" in analysis:
            print("Assistant: It seems your query is not medically related. I am specialized in providing information about medical conditions. How can I assist you with a medical question?")
            continue
        
        if "Vague Medical Query" in analysis:
            response_text = f"To give you the best information, could you describe your symptoms in more detail? For Example: {analysis.split('For Example:')[1].strip()}"
            print(f"Assistant: {response_text}")
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=response_text))
            continue

        print("Reformulating query with history...")
        standalone_query = reformulate_query_with_history(query, chat_history)
        search_query = reformulate_query_for_Search(query, chat_history)
        print(f"Database search query: {standalone_query}\n")
        print(f"Web Search query: {search_query}\n")

        print("Retrieving information from local QA database...\n")
        docs_from_qa = relevant_QA_database(standalone_query) # This returns a list of Documents
    
        
        all_docs = docs_from_qa
        
        if needs_web_search(query, docs_from_qa, chat_history):
            print("--- Web search is needed. Attempting to crawl web... ---")
            try:
                link, title, snippet = firecrawlSearch(query)
                
                if link and title and snippet:
                    print(f"--- found {link} from web search to context. ---")
                    docs_from_web = crawl_web(link, title, snippet)
                    all_docs.extend(docs_from_web)
                else:
                    print("--- Web search failed to retrieve any documents. Continuing with local documents only. ---")
            except Exception as e:
                print(f"--- ERROR: Web search failed with exception: {e} ---")
                print("--- Continuing with local documents only. ---")

        if not all_docs:
            response = "I could not find any relevant information from local or web sources. It's best to consult a medical professional for advice."
            print(f"\nAssistant:\n{response}")
        else:
            print("Generating a response based on the combined information...")
            response = generate_conversational_response(query, all_docs, chat_history)
            print(f"\nAssistant:\n{response}")

        chat_history.append(HumanMessage(content=query))
        chat_history.append(AIMessage(content=response))

if __name__ == "__main__":
    main()