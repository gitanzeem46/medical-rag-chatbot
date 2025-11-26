import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_groq import ChatGroq

# Disable LangSmith tracing (to avoid 404 errors)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_API_KEY"] = ""

# Load Groq API
# NOTE: Replace "YOUR_GROQ_API_KEY_HERE" with your actual Groq API Key
os.environ["GROQ_API_KEY"] = "YOUR_GROQ_API_KEY_HERE"

# Initialize LLM
llm = ChatGroq(model="mixtral-8x7b-32768")

# Updated Local Prompt - No Hub Usage
prompt = PromptTemplate(
    input_variables=["context", "input"],
    template="""
You are a helpful medical AI assistant.
Use ONLY the given context to answer medically accurate information.
If answer is not in context, reply: "I’m not sure from given medical knowledge."

Context:
{context}

Question:
{input}

Answer:
"""
)

st.title("🩺 AI Medical RAG Chatbot")

# Load embeddings
embeddings = HuggingFaceEmbeddings()

# Load existing FAISS DB
# NOTE: You must have a 'vectorstore' directory with your FAISS index files in the same directory as this script.
try:
    db = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Error loading vectorstore: {e}. Please ensure the 'vectorstore' directory exists and contains your FAISS index.")
    st.stop()


# Query Input
user_query = st.text_input("Ask any medical question:")

if st.button("Get Answer"):
    if user_query.strip() == "":
        st.warning("Please enter a question.")
    else:
        # Retrieve documents
        docs = retriever.get_relevant_documents(user_query)
        context = "\n\n".join([doc.page_content for doc in docs])

        # Run prompt through LLM
        final_input = prompt.format(context=context, input=user_query)
        # Using invoke instead of predict for ChatGroq for better compatibility
        response = llm.invoke(final_input).content

        st.subheader("Answer:")
        st.write(response)
