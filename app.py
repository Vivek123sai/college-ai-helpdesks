import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile, os

st.set_page_config(page_title="College AI Helpdesk", page_icon="🎓")
st.title("🎓 College AI Helpdesk")

# Sidebar mein API Key daalne ki jagah
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Groq API Key daalo", type="password", placeholder="gsk_...")
    st.markdown("Free key: [console.groq.com](https://console.groq.com)")

# PDF upload karne ka option
uploaded_file = st.file_uploader("College ka PDF upload karo (notes/FAQ/rules)", type="pdf")

if uploaded_file and api_key:
    # PDF ko process karke chunks mein todna
    if "vectorstore" not in st.session_state or st.session_state.get("file_name") != uploaded_file.name:
        with st.spinner("PDF padh raha hoon..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            docs = PyPDFLoader(tmp_path).load()
            chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

            # Embeddings banao aur ChromaDB mein store karo
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings)
            os.unlink(tmp_path)

            st.session_state.vectorstore = vectorstore
            st.session_state.file_name = uploaded_file.name
            st.session_state.messages = []
        st.success("Ready! Ab sawaal pucho.")

    # Groq LLM setup
    llm = ChatGroq(api_key=api_key, model_name="llama-3.3-70b-versatile", temperature=0.1)
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

    # Prompt: Sirf PDF ke basis par jawab do
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    Question: {question}
    If the answer is not in the context, say "Ye information mere paas nahi hai."
    """)

    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

    # Chat history dikhana
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    if question := st.chat_input("College ke baare mein kuch pucho..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            response = chain.invoke(question)
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})