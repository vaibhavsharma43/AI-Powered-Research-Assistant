import streamlit as st
import requests

# Change the API_URL if your FastAPI server is running on a different host/port.
API_URL = "http://localhost:8000/search"

st.title("AI-Powered Research Assistant")
st.write("Enter a query to search for news/articles and get a summary with sentiment analysis.")

query = st.text_input("Query:")

if st.button("Search") and query:
    payload = {"query": query}
    with st.spinner("Searching, scraping, and processing content..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                if results:
                    for item in results:
                        st.subheader(f"Source URL: {item.get('url')}")
                        st.markdown("**Summary:**")
                        st.write(item.get("summary", "No summary available."))
                        st.markdown("**Sentiment Analysis:**")
                        st.write(item.get("sentiment", "No sentiment data available."))
                        st.markdown("---")
                else:
                    st.error("No results found.")
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"An error occurred: {e}")
