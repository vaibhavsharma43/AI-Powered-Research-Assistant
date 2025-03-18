from pydantic import BaseModel

class QueryInput(BaseModel):
    QueryInput: str
class ResultItem(BaseModel):
    url: str
    summary: str
    sentiment: str
class QueryResponse(BaseModel):
    urls: list[str]
    summaries: list[str]
    sentiments: list[str] 