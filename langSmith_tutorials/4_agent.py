from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import create_agent

import requests
import os
from dotenv import load_dotenv


os.environ['LANGCHAIN_PROJECT'] = 'ReAct Agent'

load_dotenv()


# --------------------------------------------------
# Tool 1: DuckDuckGo Search
# --------------------------------------------------

search_tool = DuckDuckGoSearchRun()


# --------------------------------------------------
# Tool 2: Weather
# --------------------------------------------------

@tool
def get_weather_data(city: str) -> str:
    """
    Fetches the current weather data for a given city.
    """

    url = (
        "https://api.weatherstack.com/current"
        "?access_key=f07d9636974c4120025fadf60678771b"
        f"&query={city}"
    )

    response = requests.get(url)

    return response.text


# --------------------------------------------------
# LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)


# --------------------------------------------------
# Create Agent
# --------------------------------------------------

agent = create_agent(
    model=llm,
    tools=[
        search_tool,
        get_weather_data
    ]
)


# --------------------------------------------------
# Invoke Agent
# --------------------------------------------------

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Identify the birthplace city of Rohit Sharma (search) and give its today's temperature."
        }
    ]
})


print("\n================ AGENT RESPONSE ================\n")

print(response["messages"][-1].content)

print("\n=================================================\n")