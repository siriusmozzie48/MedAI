from crawlSchema import *
from condition import *
from firecrawlSearch import *
import asyncio    
from langchain_core.documents import Document
from typing import Dict, List, Any

def crawl_web(link, title, snippet) -> List[Document]:
    # Check if web search returned valid results
    if not link or not title or not snippet:
        print("--- DEBUG: WebSearch returned empty results. Cannot proceed with crawling. ---")
        return []
    
    print(f"--- DEBUG: WebSearch successful. Link: {link[:50]}... ---")
    
    try:
        condition = detect_conditions(link, title, snippet)
        print(f"--- DEBUG: Condition detected: {condition} ---")
        
        print("---Crawling the website----\n")
        content = asyncio.run(information_from_url(link, condition))
        
        if not content or not content[0]:
            print("--- DEBUG: No content retrieved from URL crawling. ---")
            return []
            
        docs_from_web = convert_dict_to_documents(content[0])
        print(f"--- DEBUG: Converted {len(docs_from_web)} documents from web content. ---")
        return docs_from_web
        
    except Exception as e:
        print(f"--- ERROR: Exception during web crawling: {e} ---")
        import traceback
        print(f"--- ERROR: Full traceback: {traceback.format_exc()} ---")
        return []

def convert_dict_to_documents(data_dict: Dict[str, Any]) -> List[Document]:
    """
    Converts a dictionary with topic keys and detail lists into a list of Document objects.
    """
    documents = []
    # Iterate through the dictionary's keys ('symptoms', 'causes', etc.) and values (the lists of strings)
    for topic, details_list in data_dict.items():
        # Ensure the value is a list and contains items before proceeding
        if isinstance(details_list, list) and details_list:
            # Iterate through each individual detail string in the list
            for detail in details_list:
                # Create structured page content that preserves the context of the topic
                page_content = f"Topic from Web: {topic.replace('_', ' ').title()}, Details: {detail}"
                
                # Create a new Document object
                doc = Document(
                    page_content=page_content
                )
                
                # Add the newly created Document to our list
                documents.append(doc)
                
    return documents
