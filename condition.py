from langchain_google_genai import ChatGoogleGenerativeAI

client = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Using a fast and powerful Gemini model
    temperature=0.7,
    max_output_tokens=4096 
)

def detect_conditions(url: str, title:str, snippet:str):
    prompt = f"Using the string of the URL '{url}', the title of the webpage '{title}' and the snippet '{snippet}' try and identify the medical topic of the webpage by extracting the topic from the URL, if you cannot find the topic from the URL characters OR if the topic is non medical (irrelevant), simply return 'unknown'"
    return client.invoke(prompt).content  # Clinical NER model
