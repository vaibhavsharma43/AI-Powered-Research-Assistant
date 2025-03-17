"""
================================================================================
AI-Powered Research Assistant - Code Overview
================================================================================

Overview:
    This file implements an AI-powered research assistant that performs the following tasks:
    - Accepts user input queries to search for relevant news/articles using DuckDuckGo.
    - Retrieves search results and extracts valid URLs.
    - Scrapes web content from the URLs using Crawl4AI.
    - Cleans and extracts the main article text from raw HTML using BeautifulSoup and
      readability-lxml.
    - Splits long texts into manageable chunks for processing.
    - Generates summaries for the extracted content using LangChain's summarization chain.
    - Analyzes the sentiment of the content using a custom LLM chain with LangChain.

Objectives:
    - Automate the process of gathering and analyzing web-based content.
    - Provide concise summaries and sentiment insights for various topics.
    - Demonstrate the integration of multiple libraries (dotenv, pydantic, Crawl4AI, 
      LangChain, DuckDuckGo Search, BeautifulSoup, and readability-lxml) to build a 
      cohesive AI-driven application.
    - Serve as a foundation for further expansion into a fully featured research assistant,
      potentially with a web interface or API integration.

Key Components:
    - Environment Configuration: Loads API keys and configurations using python-dotenv.
    - Web Search: Retrieves search results using DuckDuckGo (via the DDGS class).
    - Content Scraping: Fetches and extracts raw HTML content using Crawl4AI.
    - HTML Cleaning: Utilizes BeautifulSoup and readability-lxml to extract main text content.
    - Text Processing: Splits long content into chunks using LangChain’s RecursiveCharacterTextSplitter.
    - Summarization & Sentiment Analysis: Uses LangChain chains (with ChatGroq) to generate
      article summaries and perform sentiment analysis.

Usage:
    - Ensure that the required environment variables (e.g., Groq API key) are set.
    - Install dependencies via the provided requirements.txt file.
    - Run the script to input a query and view summarized content along with sentiment analysis.

================================================================================
"""



import asyncio
import os
from dotenv import load_dotenv
from pydantic import SecretStr
from crawl4ai import AsyncWebCrawler
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from duckduckgo_search import DDGS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

# Load environment variables
load_dotenv()
api_key = os.getenv("Groq_API_KEY")
if not api_key:
    raise ValueError("Groq_API_KEY not found in environment variables")

# Initialize Groq LLM (consider specifying model explicitly if needed)
llm = ChatGroq(api_key=SecretStr(api_key), temperature=0.3)

# Utility: Use readability-lxml to extract the main article content and then clean it with BeautifulSoup
def extract_main_text(html):
    try:
        # First try with readability
        readable_article = ReadabilityDocument(html)
        summary_html = readable_article.summary()
        
        # Enhanced BeautifulSoup cleaning
        soup = BeautifulSoup(summary_html, "html.parser")
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer', 'iframe', 'noscript', 'aside']):
            element.decompose()
            
        # Look for common article containers
        article = (
            soup.find('article') or 
            soup.find(class_=lambda x: x and ('article' in x.lower() or 'content' in x.lower())) or
            soup.find(id=lambda x: x and ('article' in x.lower() or 'content' in x.lower()))
        )
        
        if article:
            # Use the article content if found
            text = ' '.join(p.get_text().strip() for p in article.find_all('p') if p.get_text().strip())
        else:
            # Fallback to all paragraphs if no article container found
            text = ' '.join(p.get_text().strip() for p in soup.find_all('p') if p.get_text().strip())
            
        # Clean up the text
        text = ' '.join(text.split())  # Remove extra whitespace
        return text if text else "No content found"
        
    except Exception as e:
        print(f"Content extraction failed: {e}")
        # Basic fallback
        soup = BeautifulSoup(html, "html.parser")
        for script in soup(['script', 'style', 'nav', 'header', 'footer']):
            script.decompose()
        return ' '.join(soup.stripped_strings)

# Step 1: Get search links using DDGS as a context manager
def get_search_links(query, num_results=5):
    urls = []
    with DDGS() as searcher:
        results = searcher.text(query, max_results=num_results)
        for result in results:
            link = result.get("href", "")
            if link.startswith("http"):
                urls.append(link)
    return urls

# Step 2: Scrape content from the links using Crawl4AI
async def fetch_content(url):
    async with AsyncWebCrawler() as crawler:
        try:
            result = await crawler.arun(url=url)
            html_content = str(result)
            # Extract the main text using readability
            cleaned_text = extract_main_text(html_content)
            return cleaned_text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return ""

async def fetch_all_content(urls):
    tasks = [fetch_content(url) for url in urls]
    results = await asyncio.gather(*tasks)
    # Filter out empty results
    return [res for res in results if res]

# Step 3: Summarize extracted content using LangChain
def summarize_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    summaries = []
    for chunk in chunks[:3]:  # Process up to first 3 chunks
        docs = [Document(page_content=chunk)]
        chain = load_summarize_chain(llm, chain_type="stuff")
        try:
            summary = chain.run(docs)
            summaries.append(summary)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
    return " ".join(summaries)

# Step 4: Analyze sentiment using LLM with a custom prompt
prompt = PromptTemplate(
    input_variables=["text"],
    template="Analyze the sentiment of the following text: {text}"
)
sentiment_chain = LLMChain(llm=llm, prompt=prompt)

def analyze_sentiment(text):
    # If text is too long, analyze only the first chunk
    if len(text) > 2000:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        text = chunks[0]
    return sentiment_chain.run(text=text)

# Main async function to integrate all steps
async def main():
    query = input("Enter your search query: ")
    urls = get_search_links(query, num_results=5)
    
    if not urls:
        print("No valid URLs found for the query.")
        return

    print("\nFound URLs:")
    for url in urls:
        print(url)

    print("\nFetching content from URLs...")
    scraped_data = await fetch_all_content(urls)

    for i, content in enumerate(scraped_data):
        print(f"\n--- Processing content from: {urls[i]} ---")
        summary = summarize_text(content)
        sentiment = analyze_sentiment(content)
        print("\nSummary:\n", summary)
        print("\nSentiment Analysis:\n", sentiment)

if __name__ == "__main__":
    asyncio.run(main())
