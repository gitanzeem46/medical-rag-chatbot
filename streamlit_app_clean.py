import os
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# ----------- CONFIG ----------------
DB_FAISS_PATH = "vectorstore"   # <-- Correct folder only


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt


def main():
    st.title("🧠 AI Medical RAG Chatbot")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    query = st.chat_input("Ask your medical question...")

    if query:

        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        try:
            vectorstore = get_vectorstore()

            if vectorstore is None:
                st.error("❌ Failed to load vectorstore")
                return

            # ----------- LLM -------------
            GROQ_API_KEY = os.getenv("GROQ_API_KEY")

            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
                api_key=GROQ_API_KEY
            )

            # ----------- Prompt from Hub -------------
            retrieval_prompt = hub.pull("langchain-ai/rag-prompt")

            # ----------- Create Chain -------------
            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_prompt)

            rag_chain = create_retrieval_chain(
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                combine_documents_chain=combine_docs_chain,
            )

            response = rag_chain.invoke({"input": query})

            result = response["output_text"]  # <-- New correct key

            st.chat_message("assistant").markdown(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

        except Exception as e:
            st.error(f"⚠ ERROR: {str(e)}")


if __name__ == "__main__":
    main()
