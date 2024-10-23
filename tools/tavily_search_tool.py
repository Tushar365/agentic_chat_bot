from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults
import os

def initialize_tavily_tool():
    """Initializes and returns a TavilySearchResults tool.

    Args:
        api_key (str, optional): The Tavily API key. If not provided, 
                                  it will attempt to retrieve it from 
                                  the environment variable "TAVILY_API_KEY".
    
    Returns:
        TavilySearchResults: The initialized Tavily search tool.
        
    Raises:
        ValueError: If the API key is not provided and cannot be found 
                    in the environment variables.
    """
    os.environ["TAVILY_API_KEY"] = os.getenv('TAVILY_API_KEY')
    search = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(api_wrapper=search)
    return tavily_tool