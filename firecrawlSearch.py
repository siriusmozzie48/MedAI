from firecrawl import FirecrawlApp
import os


def firecrawlSearch(query):
    app = FirecrawlApp(api_key=os.environ.get("Firecrawl_API_KEY"))
    print(f"DEBUG: Query sent: {query}")
    search_result = app.search(query, limit=1)
    print(f"DEBUG: Number of results: {len(search_result.data)}")

    if not search_result.data:
        print("No search results found.")
        return None, None, None

    link = search_result.data[0]['url']
    title = search_result.data[0]['title']
    snippet = search_result.data[0]['description']

    # for result in search_result.data:
    #     print(f"Title: {result['title']}")
    #     print(f"URL: {result['url']}")
    #     print(f"Description: {result['description']}") 

    return link, title, snippet