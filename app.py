import nltk # natural language toolkit
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
import streamlit as st
import openai
import os

# Loading API, Embeddings, Chroma

api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Setting background image

def set_background(image_url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("https://collectionapi.metmuseum.org/api/collection/v1/iiif/338737/779131/main-image")

#Customization of text

st.markdown(
    """
    <style>
    .title-style {
        background-color: rgba(0, 0, 0, 0.7);
        padding: 10px;
        border-radius: 8px;
        color: antiquewhite;
        font-size: 40px;
    }

    .header-style {
        background-color: rgba(0, 0, 0, 0.7);
        color: antiquewhite;
        font-size: 20px;
        font-weight: normal;
        text-align: start;
        padding: 10px;
        border-radius: 8px;
    }

    .text-style {
        background-color: rgba(0, 0, 0, 0.7);
        color: antiquewhite;
        font-size: 20px;
        font-weight: normal;
        text-align: start;
        padding: 10px;
        border-radius: 8px;
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title

title = "Piranesi by Sussanna Clarke Q&A"
st.markdown(f'<div class="title-style">{title}</div>', unsafe_allow_html=True)

# Input

st.markdown('<div class="header-style">Ask me anything about the book:</div>', unsafe_allow_html=True)
user_question = st.text_input("")


retrieved_docs = db.similarity_search(user_question, k=10)

def _get_document_prompt(docs):
    prompt = "\n"
    for doc in docs:
        prompt += "\nContent:\n"
        prompt += doc.page_content + "\n\n"
    return prompt

# Generate a formatted context from the retrieved documents

formatted_context = _get_document_prompt(retrieved_docs)

# ChatBot Prompt

prompt = f"""
## SYSTEM ROLE
You are an expert on the book 'Piranesi' by Sussanna Clarke. Your job is to:
-Answer the user's question directly and without adding introductory sentences that name the book or author.
-Recognize characters, places, objects and events in the book.
-Provide detailed explanations of characters, locations and themes.
-Analyze and corelate: notice relationships between characters (e.g., when one character has multiple identities or names) and explain them.
-Answer questions about the book, including character motivations and plot points.
-Provide summaries of chapters or sections of the book.
-Explain the significance of specific quotes or passages.
-Clarify and interpret: if a user asks about symbolic meanings or hidden themes, provide analyses and interpretations.
-Provide context: if a user asks about a specific event or character, provide background information to help them understand the significance of that event or character.
If unsure, say 'I don't know' and adjuct what you think the answer is.

## USER QUESTION
The user has asked: 
"{user_question}"

## CONTEXT
Here is the relevant content from the book:  
'''
{formatted_context}
'''

## GUIDELINES
1. **Accuracy**:
   - Character recognition: provide full identity of the characters (including alternate names and why they have it), their role in the story, and any major developments.
   - Place recognition: provide full names of places, their significance in the story, and any major events that occur there.
   - If the answer cannot be found, explicitly state: "I couldn't find the answer in the book", and ask the user to provide more context or details.
   - Provide page numbers for any specific quotes or passages you reference in your answer.
   - Warn the user about speculative content: if you are unsure about a specific detail, make it clear that you are speculating and provide your reasoning.

2. **Transparency**:   
   - Warn the user about speculative content: if you are unsure about a specific detail, make it clear that you are speculating and provide your reasoning. 
   - Spoiler sensitivity: if the answer involves explaining the development of the story, warn the user before reavealing the information.

3. **Clarity**:  
   - Your response should be clear and easy to understand.

## TASK 
1. Point the user to relevant parts of the book.  
2. Provide the response in the following format:

## RESPONSE FORMAT
'''
[Brief Title of the Answer]

[Answer in simple, clear text.]
'''
**Source**:  
• [Book Title], Page(s): [...]
"""

client = openai.OpenAI()
model_params = {
    'model': 'gpt-4o',
    'temperature': 0.7,  # Increase creativity
    'max_tokens': 4000,  # Allow for longer responses
    'top_p': 0.9,        # Use nucleus sampling
    'frequency_penalty': 0.5,  # Reduce repetition
    'presence_penalty': 0.6    # Encourage new topics
}

messages = [{'role': 'user', 'content': prompt}]
completion = client.chat.completions.create(messages=messages, **model_params, timeout=120)
answer = completion.choices[0].message.content

# Answer styling
st.markdown(f'<div class="text-style"><br><b>Answer:</b>{answer}</div>', unsafe_allow_html=True)

st.markdown('<div class="header-style">Image: Giovanni Battista Piranesi </div>', unsafe_allow_html=True)