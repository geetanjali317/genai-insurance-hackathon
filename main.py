# main.py

import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI

from core.correlation import (
    find_research_references_correlating_with_each_news_snnipets,
)
from core.data_structuring import structure_the_response
from core.visualizer import define_ui_and_visual_elements

# Load environment variables
load_dotenv()
openai_apikey = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

# Init LLM + Tools
model = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=openai_apikey)

tool = TavilySearchResults(
    tavily_api_key=tavily_key,
    max_results=10,
    include_answer=True,
    include_raw_content=True,
)

# Query Example
response = tool.invoke({"query": "climate insurance risk 2025"})

# Structure, correlate, and visualize
structured_news = structure_the_response(response)
enriched_responses, references_dict = (
    find_research_references_correlating_with_each_news_snnipets(structured_news)
)
dashboard_data = define_ui_and_visual_elements(enriched_responses, references_dict)

# -------------------------
# Streamlit App UI
# -------------------------


def filter_by_tag(data, selected_tag):
    return [item for item in data if selected_tag in item["tags"]]


def get_all_tags(data_list):
    tags = set()
    for item in data_list:
        tags.update(item["tags"])
    return sorted(tags)


st.title("🌍 Climate Risk Insurance Dashboard")

# Sidebar tag selection
all_tags = get_all_tags([item for item in dashboard_data])
selected_tag = st.sidebar.selectbox("Filter by Tag", all_tags)

# Display Articles + Research
st.header(f"News Tagged: {selected_tag}")
for item in dashboard_data:
    if selected_tag in item["tags"]:
        st.subheader(item["title"])
        st.write(item["summary"])
        st.caption("Tags: " + ", ".join(item["tags"]))
        if item["related_papers"]:
            with st.expander("🔬 Related Research"):
                for paper in item["related_papers"]:
                    st.markdown(
                        f"**{paper['title']}** ({paper['date']})  \n{paper['authors']}  \n{paper['abstract']}"
                    )
        st.markdown("---")

st.sidebar.markdown("----")
st.sidebar.info("Powered by Tavily + LangChain + OpenAI")
# st.title("Climate Risk Insurance Dashboard")
# st.write("App loaded successfully")
