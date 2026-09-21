import streamlit as st
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page configuration
st.set_page_config(page_title="Hasamex Transcript Analyzer", layout="wide")
st.title("🎙️ Hasamex Expert Call Transcript Analyzer (Local & Private)")
st.info("💡 Running 100% locally using Ollama. No API keys or external data sharing required.")

# Initialize session state for vectorstore
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

def load_and_process_documents():
    """Loads transcripts and creates a FAISS vector store for RAG."""
    files = [
        "data/Transcript_1_France.txt", 
        "data/Transcript_2_Germany.txt", 
        "data/Transcript_3_UK.txt"
    ]
    docs = []
    for file in files:
        if os.path.exists(file):
            loader = TextLoader(file, encoding="utf-8")
            docs.extend(loader.load())
        else:
            st.warning(f"⚠️ Missing: {file}. Please download the case pack and place it in the 'data' folder.")
            return None
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # Use local Ollama embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = FAISS.from_documents(splits, embeddings)
    st.session_state.vectorstore = vectorstore
    return vectorstore

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

st.markdown("---")

# FEATURE 1: Interview Guide Analysis
st.header("1. 📋 Interview Guide Analysis")
st.markdown("Answers the core interview guide questions with exact quotes and timestamps.")
if st.button("Analyze Interview Guide"):
    with st.spinner("Analyzing transcripts locally (this may take 10-15 seconds)..."):
        vectorstore = load_and_process_documents()
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            # Use local Llama 3.2 model
            llm = ChatOllama(model="llama3.2", temperature=0) 
            
            guide_prompt = ChatPromptTemplate.from_template("""
            You are an expert analyst. Based on the provided transcript excerpts, answer the interview guide questions.
            CONSTRAINTS:
            1. Provide a concise, direct answer.
            2. You MUST include at least one EXACT quote from the transcripts to support your answer.
            3. You MUST include the supporting timestamp for each quote.
            
            Context: {context}
            """)
            
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | guide_prompt
                | llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke("Answer the interview guide questions based on the context.")
            st.markdown(response)

# FEATURE 2: Themes & Disagreements
st.markdown("---")
st.header("2. 🔍 Common Themes & Disagreements")
st.markdown("Identifies cross-call patterns and conflicting viewpoints.")
if st.button("Identify Themes & Disagreements"):
    with st.spinner("Analyzing cross-transcript themes locally..."):
        all_text = ""
        for file in ["data/Transcript_1_France.txt", "data/Transcript_2_Germany.txt", "data/Transcript_3_UK.txt"]:
            if os.path.exists(file):
                with open(file, "r", encoding="utf-8") as f:
                    all_text += f"\n--- {os.path.basename(file)} ---\n" + f.read()
        
        if all_text:
            llm = ChatOllama(model="llama3.2", temperature=0)
            theme_prompt = ChatPromptTemplate.from_template("""
            You are an expert market analyst. Analyze the following expert call transcripts.
            Identify:
            1. Top 3 Common Themes across all calls.
            2. Key Disagreements or conflicting viewpoints between the experts.
            
            STRICT CONSTRAINTS:
            - For EVERY theme and disagreement, you MUST provide an EXACT quote from the transcript.
            - You MUST include the supporting timestamp for each quote.
            
            Transcripts:
            {all_text}
            """)
            
            chain = theme_prompt | llm | StrOutputParser()
            response = chain.invoke({"all_text": all_text})
            st.markdown(response)

# FEATURE 3: Q&A Across Transcripts
st.markdown("---")
st.header("3. 💬 Ask Questions Across Transcripts")
user_query = st.text_input("Ask a specific question about the transcripts:")
if user_query:
    with st.spinner("Searching transcripts locally..."):
        vectorstore = load_and_process_documents()
        if vectorstore:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
            
            qa_prompt = ChatPromptTemplate.from_template("""
            Answer the user's question based ONLY on the provided transcript excerpts.
            CONSTRAINTS:
            - Always include the EXACT quote from the transcript to support your answer.
            - Always include the supporting timestamp.
            - If the answer is not in the context, state: "I cannot find this information in the transcripts."
            
            Question: {input}
            Context: {context}
            """)
            
            llm = ChatOllama(model="llama3.2", temperature=0)
            
            rag_chain = (
                {"context": retriever | format_docs, "input": RunnablePassthrough()}
                | qa_prompt
                | llm
                | StrOutputParser()
            )
            
            response = rag_chain.invoke(user_query)
            st.markdown(response)