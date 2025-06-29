import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from crawl4ai import AsyncWebCrawler, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

class MedicalInfo(BaseModel):
    symptoms: Optional[List[str]] = Field(description=f"A list of specific symptoms mentioned on the page.")
    causes: Optional[List[str]] = Field(description=f"A list of potential causes mentioned on the page.")
    treatments: Optional[List[str]] = Field(description=f"A list of recommended treatments, remedies, or management strategies.")
    diagnosis: Optional[List[str]] = Field(description=f"A list of diagnostic methods or tests mentioned.")
    prevention: Optional[List[str]] = Field(description=f"A list of preventative measures described on the page.")
    when_to_see_doctor: Optional[List[str]] = Field(description="A list of red flags or specific advice on when to seek medical help.")
    emergency_situations: Optional[List[str]] = Field(description="A list of symptoms or situations described as requiring emergency medical attention.")

async def information_from_url(url: str, condition: str) -> Optional[MedicalInfo]:
    """
    Crawls a URL to extract structured medical information related to a specific condition.
    Returns a Pydantic object containing the structured data.
    """
    instruction = (
        f"You are an expert medical data analyst. From the content of the webpage, "
        f"extract all relevant information about the medical condition '{condition}'. "
        f"Format your output strictly according to the provided JSON schema. "
        f"Focus only on information present in the text."
    )
    
    extraction_strategy = LLMExtractionStrategy(
        provider="gemini/gemini-1.5-flash",
        api_token=os.environ.get("GOOGLE_API_KEY"),
        schema=MedicalInfo.model_json_schema(),
        extraction_type="schema",
        instruction=instruction 
    )

    async with AsyncWebCrawler(verbose=True) as crawler:
        result = await crawler.arun(
            url=url,
            extraction_strategy=extraction_strategy,
            cache_mode=CacheMode.BYPASS
        )
        data_str = result.extracted_content
        data = json.loads(data_str)
        return data