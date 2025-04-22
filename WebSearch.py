import requests
API_KEY = "AIzaSyBRW906nm5qDN8CzZII6Uhq5TFkVl9g8i4"
SEARCH_ENGINE_ID = "407d9ce84cf1c4615"
def WebSearch(search_query):

    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q':search_query,
        'key':API_KEY,
        'cx':SEARCH_ENGINE_ID
    }
    link = {}
    response = requests.get(url, params=params)
    results = (response.json())
    if 'items' in results :
        link = results['items'][0]['link']
        title = results['items'][0]['title']
        snippet = results['items'][0]['snippet']
    return link, title, snippet





