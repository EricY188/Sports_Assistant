"""
Sports Expert Chatbot with Streamlit UI
Integrates RAG and OpenAI GPT-4
"""

import streamlit as st
import openai
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# 1. Model Initialization
@st.cache_resource
def initialize_components():
    """Initialize AI components"""
    # Set up OpenAI
    openai.api_key = os.getenv("OPENAI_API_KEY") or st.secrets["OPENAI_API_KEY"]
    
    # Initialize embeddings
    embeddings = OpenAIEmbeddings()
    
    # Initialize knowledge base
    sports_knowledge = [
        "NBA playoff rules 2023...",
        "Football offside rule...",
        "Marathon training principles...",
    ]
    knowledge_base = FAISS.from_texts(sports_knowledge, embeddings)
    
    return embeddings, knowledge_base

# 2. Prompt Engineering
def create_sports_prompt():
    """Construct system prompt template"""
    return PromptTemplate(
        input_variables=["context", "question"],
        template="""You are SportGPT, a professional sports assistant with expertise in:
1. Game rules interpretation
2. Training plan formulation  
3. Real-time match analysis

Context: {context}
Current Date: {current_date}

Question: {question}

Response requirements:
■ Cite data sources
■ Provide step-by-step explanations
■ Highlight safety considerations
Answer in English with proper formatting:"""
    )

# 3. Main Application
def main():
    # Initialize components
    embeddings, knowledge_base = initialize_components()
    
    # Configure page
    st.set_page_config(page_title="SportGPT", page_icon="🏀")
    st.title("🏆 SportGPT - Sports Assistant")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        openai_key = st.text_input("OpenAI API Key", type="password")
        temperature = st.slider("Creativity Level", 0.0, 1.0, 0.3)
        max_tokens = st.number_input("Max Response Length", 100, 2000, 500)
        
        if openai_key:
            openai.api_key = openai_key
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input handling
    if user_input := st.chat_input("Enter your sports question:"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.spinner("Analyzing..."):
            try:
                # Retrieve relevant context
                docs = knowledge_base.similarity_search(user_input, k=3)
                context = "\n".join([d.page_content for d in docs])
                
                # Create prompt
                prompt_template = create_sports_prompt()
                formatted_prompt = prompt_template.format(
                    context=context,
                    question=user_input,
                    current_date="2024-05-20"
                )
                
                # Generate response using OpenAI
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": formatted_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens
                ).choices[0].message.content
                
            except Exception as e:
                response = f"Error: {str(e)}"
        
        # Display response
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
