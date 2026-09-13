import streamlit as st
from google import genai
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Read API key
api_key = ""

# Create Gemini client
client = genai.Client(api_key=api_key)


# Streamlit UI
st.title("Gemini LLM App")
st.header("Summarization header")

prompt = st.text_area("Enter your prompt")


if st.button("Generate"):

    if not prompt:
        st.warning("Please enter a prompt.")
    else:

        response = client.models.generate_content(
            model="gemini-3.5-flash-lite",
            contents=prompt
        )
        print(dir(response))

        st.write(response.text)


###############################################################################################
from dotenv import load_dotenv
from openai import OpenAI

# Load variables from .env
load_dotenv()
print(load_dotenv())

# OpenAI automatically reads OPENAI_API_KEY
client = OpenAI()

response = client.responses.create(
    model="gpt-5.6-luna",
    # model="gpt-4.1",
    input="Explain Generative AI in simple words.",
    text={
        "format": {
            "type": "text"
        },
        "verbosity": "medium"
    },
    reasoning={
        "effort": "medium",
        "summary": "auto"
    }
)

print(response.output_text)
