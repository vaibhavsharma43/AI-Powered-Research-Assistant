# test/test_main.py
import pytest
import asyncio

# Import functions from your main project file.
# It is assumed your main code is in a file named "main.py" in the project root.
from main import (
    extract_main_text,
    get_search_links,
    summarize_text,
    analyze_sentiment,
    fetch_content,
    fetch_all_content,
)

# ------------------------------------------------------------------------------
# Test for extract_main_text
# ------------------------------------------------------------------------------
def test_extract_main_text_readability():
    # HTML with an article container
    html = (
        "<html><head><title>Test</title></head>"
        "<body><article><p>Hello World!</p></article></body></html>"
    )
    text = extract_main_text(html)
    assert "Hello World!" in text, "Should extract the article's paragraph text."

def test_extract_main_text_fallback():
    # HTML without an explicit article container
    html = (
        "<html><head><title>Test</title></head>"
        "<body><p>Just a simple paragraph.</p></body></html>"
    )
    text = extract_main_text(html)
    assert "Just a simple paragraph." in text, "Fallback extraction should work."

# ------------------------------------------------------------------------------
# Test for get_search_links (integration test: requires network)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_get_search_links():
    query = "AI trends"
    urls = get_search_links(query, num_results=3)
    assert isinstance(urls, list), "Should return a list of URLs."
    for url in urls:
        assert url.startswith("http"), "Each URL should start with 'http'."

# ------------------------------------------------------------------------------
# Test for summarize_text
# ------------------------------------------------------------------------------
def test_summarize_text():
    sample_text = (
        "Artificial intelligence is revolutionizing industries. "
        "It is used in many applications, from healthcare to finance."
    )
    summary = summarize_text(sample_text)
    assert isinstance(summary, str), "Summary should be a string."
    assert len(summary) > 0, "Summary should not be empty."

# ------------------------------------------------------------------------------
# Test for analyze_sentiment
# ------------------------------------------------------------------------------
def test_analyze_sentiment():
    sample_text = "Artificial intelligence is revolutionizing industries."
    sentiment = analyze_sentiment(sample_text)
    assert isinstance(sentiment, str), "Sentiment analysis result should be a string."
    assert len(sentiment) > 0, "Sentiment result should not be empty."

# ------------------------------------------------------------------------------
# Async tests for content fetching using Crawl4AI (integration tests)
# ------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_fetch_content():
    # Use a known URL (integration test; requires network)
    url = "https://techcrunch.com/category/artificial-intelligence/"
    content = await fetch_content(url)
    assert isinstance(content, str), "Fetched content should be a string."
    assert len(content) > 0, "Fetched content should not be empty."

@pytest.mark.asyncio
async def test_fetch_all_content():
    # Use a list of known URLs (integration test; requires network)
    urls = [
        "https://techcrunch.com/category/artificial-intelligence/",
        "https://apnews.com/",
    ]
    contents = await fetch_all_content(urls)
    assert isinstance(contents, list), "Fetched contents should be a list."
    # At least one URL should return non-empty content
    assert any(len(content) > 0 for content in contents), "At least one fetched content should not be empty."
