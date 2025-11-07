import os
from dotenv import load_dotenv
import streamlit as st

from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    ("user", "{text}"),
])

# Streamlit UI
st.title("💬 Simple Chat App with HuggingFace")
input_text = st.text_input("Ask me anything:")

# Load Hugging Face model (example: google/gemma-1.1-2b-it)
llm = HuggingFaceHub(
    repo_id="google/gemma-1.1-2b-it",  # You can change this to any HF text-generation model
    model_kwargs={"temperature": 0.7, "max_new_tokens": 256}
)

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
