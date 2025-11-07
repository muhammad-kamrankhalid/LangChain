import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables (optional if using LangSmith)
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# Optional: LangSmith tracking (you can remove if not using)
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACKING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Gemma Chat App")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    ("user", "{text}"),
])

# Streamlit UI
st.title("💬 Simple Chat App with Ollama (Gemma3:1b)")
input_text = st.text_input("Ask me anything:")

# Load local Ollama model
llm = Ollama(model="gemma3:1b")

# Output parser
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Generate response
if input_text:
    with st.spinner("Thinking... "):
        response = chain.invoke({"text": input_text})
    st.write("### Response:")
    st.write(response)
