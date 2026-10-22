from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import PydanticOutputParser

from pydantic import BaseModel
from typing import List, Optional

st.set_page_config(page_title="Movie Info Extractor", page_icon="🎬", layout="centered")


@st.cache_resource
def get_model():
    return init_chat_model("google_genai:gemma-4-26b-a4b-it", temperature=0.0)


model = get_model()


class Movie(BaseModel):
    title: str
    release_date: Optional[str]
    genre: List[str]
    revenue: Optional[float]
    director: Optional[List[str]]
    lead_cast: List[str]
    runtime: Optional[str]
    rating: float


parser = PydanticOutputParser(pydantic_object=Movie)

system_prompt = """
You are a movie information extractor. The user will give you a description about a movie. Extract only the important details and return them in a concise, structured format.

Always extract these fields when present: {movie_output_format}

If a field is not mentioned in the description, write "Not mentioned" for it. Do not add information that is not in the description. Do not include any other commentary or long explanations.
"""

prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", """Movie Description:\n{description}""")]
)

print("- - " * 50)
description = input("Movie description: ")
print("_ _ " * 50)

prompt = prompt_template.invoke(
    {
        "movie_output_format": parser.get_format_instructions(),
        "description": description,
    }
)


response = model.invoke(prompt)

print(" - " * 50)
# print(response.content[1]["text"])
print(parser.parse(response.content[1]['text']))
print(" _ " * 50)
