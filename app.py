import os
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

# -------------------------
# Load Gemini API key from Streamlit secrets
# -------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# App config + header
# -------------------------
st.set_page_config(page_title="Travel Assistant RAGBot", page_icon="🌍", layout="centered")
st.title("🌍 Travel Assistant (RAG-powered)")
st.write("Upload a travel guide/itinerary (PDF). The assistant will answer based on the uploaded document.")

# -------------------------
# PDF upload
# -------------------------
uploaded_pdf = st.file_uploader("📄 Upload a travel-related PDF", type=["pdf"])
if not uploaded_pdf:
    st.info("Upload a PDF to begin.")
    st.stop()

pdf_path = os.path.join(".", uploaded_pdf.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded_pdf.read())
st.success(f"Uploaded: {uploaded_pdf.name}")

# -------------------------
# Extract text from PDF
# -------------------------
text = ""
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
st.info(f"Extracted {len(text)} characters from the PDF.")

# -------------------------
# Split into chunks
# -------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)
chunks = splitter.split_text(text)
st.write(f"📚 Document split into {len(chunks)} chunks.")

# -------------------------
# Embed chunks & store in Chroma
# -------------------------
embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
vectorstore = Chroma.from_texts(
    texts=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)
vectorstore.persist()
st.success("✅ Document embedded and stored in vector DB (Chroma).")

# -------------------------
# User query
# -------------------------
st.subheader("💬 Ask about the uploaded document")
user_query = st.text_input("Enter your question here:")

if st.button("Get Answer") and user_query:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.get_relevant_documents(user_query)

    if not relevant_docs:
        st.warning("No relevant chunks found in the document.")
    else:
        context = "\n\n---\n\n".join([f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(relevant_docs)])

        prompt = f"""
You are a friendly travel assistant. Use ONLY the information in the provided context to answer the user's question.
If the answer is not in the context, say: "I cannot answer this based on the provided context."

Context:
{context}

Question: {user_query}

Answer:
"""
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)
        response = llm.invoke(prompt)
        final_answer = response.content if hasattr(response, "content") else str(response)

        st.markdown("### 🧭 Answer:")
        st.write(final_answer)

        with st.expander("View retrieved document chunks"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
