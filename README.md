AI-Powered Research Assistant

Overview

This project integrates FastAPI (backend) and Streamlit (frontend) to build an AI-powered research assistant that fetches, summarizes, and analyzes search results using web scraping and NLP techniques.

Features

🖥 FastAPI Backend: Handles search queries and processes data.

🎨 Streamlit Frontend: Provides a user-friendly interface.

🔎 Web Scraping: Fetches relevant articles.

📝 Summarization: Extracts key information.

📊 Sentiment Analysis: Evaluates content tone.

Tech Stack

Backend: FastAPI, crawl4ai, LangChain, DuckDuckGo Search, BeautifulSoup, Readability-LXML

Frontend: Streamlit

Deployment: Uvicorn for API, Streamlit for UI

Installation

1️⃣ Clone the Repository

git clone https://github.com/your-repo/AI-Research-Assistant.git
cd AI-Research-Assistant

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Run FastAPI Backend

uvicorn main:app --reload

4️⃣ Run Streamlit Frontend

streamlit run frontend.py


{
  "results": [
    {
      "url": "https://example.com/article",
      "summary": "This article discusses the latest AI trends...",
      "sentiment": "Positive"
    }
  ]
}

Usage

1.Enter a search query in Streamlit.

2.Click 'Search' to fetch and process articles.

3.View results with summaries and sentiment analysis.



AI-Research-Assistant/
│── main.py          # FastAPI backend
│── frontend.py      # Streamlit frontend
│── requirements.txt # Dependencies
│── README.md        # Documentation

Contributing

Pull requests are welcome! Please follow best practices and submit issues if you find any bugs.
