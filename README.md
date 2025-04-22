Medical diagnosis specialist assistant using Retrieval Augmented Generation Ankara, Turkey
Personal Project Jan 2025 ‑ Present
• So far developed a parallel RAG architecture that combines pre-prepared Medical Q/A data and real-time web scraped data
according to the user query to extract and summarize retrieved documents in the desired format that relate to the medical
condition as explained by the patient.
• Working on implementing an interactive conversational loop that terminates only when the appropriate medical condition is successfully
identified.
• Technical Skills: LangChain, RAG, Prompt Engineering, Web Scraping, Python

Challenge - How many pages to scrape? Can scraping be in sync with the chat? it will slow user interface but how much

Challenge - What should be the limit of the medical content to cover? Just symptoms? or also treatments and other things? And how comprehensive should be the knowledge?  

Challenge - If condition is not specified in the scraping function, how can the relevant content be scraped still?

Challenge - Relevant content is scraped but its not comprehensive when condition is not specified - Pass url to LLM and have it decide the condition

Challenge - How to have the LLM get back and ask more questions to understand the situation?

Challenge - How to reduce hallucination? It is basically giving back some stuff not in the webpage.

Challenge - The current dataset is full of medical jargon and does not have answers to simple queries based on simple common diseases.

Challenge - Wouldn't a reasoning model be better for such an application? Well one part of it , maybe the initial part where the user just provided symptoms
