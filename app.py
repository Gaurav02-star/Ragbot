# app.py
import os
import time
import logging
from typing import List

import requests
import streamlit as st
import pdfplumber

# LangChain + Google Gemini embedding/chat libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

# LangChain agent + Tool helpers
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_ragbot")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Travel Assistant RAGBot", page_icon="🌍", layout="centered")
st.title("🌍 Travel Assistant (RAG + Web Search + Weather)")
st.write("Your AI-powered travel companion — Gemini + LangChain + OpenWeather")

# -------------------------
# Load secrets from Streamlit
# -------------------------
if "GOOGLE_API_KEY" not in st.secrets or "OPENWEATHER_API_KEY" not in st.secrets:
    st.error("❌ Please configure GOOGLE_API_KEY and OPENWEATHER_API_KEY in Streamlit Secrets.")
    st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]

# Keep env variables for libraries that expect them
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# Constants / Persistence
# -------------------------
PERSIST_DIR = "./chroma_db"  # ensure this path is writeable in your deployment
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHAT_MODEL_NAME = "gemini-2.0"  # adapt to available model in your account
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# -------------------------
# Utility: HTTP requests with retries
# -------------------------
def requests_get_with_retries(url, params=None, headers=None, retries=3, backoff=1.0, timeout=6):
    """Simple GET with retries and exponential backoff."""
    for i in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning("Request error (attempt %d/%d): %s", i + 1, retries, e)
            if i == retries - 1:
                raise
            time.sleep(backoff * (2 ** i))

# -------------------------
# Tool: Weather
# -------------------------
def weather_tool(city: str) -> str:
    """Return a short weather summary for a city using OpenWeatherMap."""
    city = city.strip()
    if not city:
        return "Please provide a city name."

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": OPENWEATHER_API_KEY, "units": "metric"}
        r = requests_get_with_retries(url, params=params)
        data = r.json()
        if data.get("cod") != 200:
            return f"City not found: {city}"
        weather = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        wind = data["wind"]["speed"]
        return f"{city.title()}: {weather}, {temp}°C, Humidity {humidity}%, Wind {wind} m/s"
    except Exception as e:
        logger.exception("Weather tool failed: %s", e)
        return f"Weather service error: {e}"

# -------------------------
# Tool: Web Search (DuckDuckGo wrapper)
# -------------------------
def web_search_tool(query: str) -> str:
    """Run a DuckDuckGoSearchRun search and return a concise combined string."""
    try:
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        return results if isinstance(results, str) else str(results)
    except Exception as e:
        logger.exception("Web search failed: %s", e)
        return f"Web search error: {e}"

# Wrap tools for LangChain Agent
web_tool = Tool(
    name="web_search",
    func=web_search_tool,
    description="Search the web for travel information. Input should be a query string."
)
weather_tool_wrapped = Tool(
    name="weather",
    func=weather_tool,
    description="Return weather for a city. Input is a city name."
)

# -------------------------
# Helper: create/load persistent Chroma vectorstore
# -------------------------
def build_or_load_vectorstore_from_chunks(chunks: List[str]):
    """Create or load a persisted Chroma vectorstore for the provided chunks."""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        # If persisted data exists, try to open
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            logger.info("Loading existing Chroma vectorstore from %s", PERSIST_DIR)
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        else:
            logger.info("Creating new Chroma vectorstore at %s", PERSIST_DIR)
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=PERSIST_DIR)
            # persist to disk so future runs don't re-embed
            try:
                vectorstore.persist()
            except Exception as e:
                logger.warning("Chroma persist() failed: %s", e)
        return vectorstore
    except Exception as e:
        logger.exception("Failed to build/load vectorstore: %s", e)
        raise

# -------------------------
# Small helper: split text into chunks with metadata
# -------------------------
def split_text_with_meta(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# -------------------------
# LangChain LLM helper (Gemini chat wrapper)
# -------------------------
def create_chat_llm(temperature=0.3):
    return ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=temperature)

# -------------------------
# Streamlit: Choose mode
# -------------------------
option = st.radio(
    "How can I assist you today?",
    ("Ask using a travel document", "Search travel info on web", "Check weather in a city", "Agent (combined tools)")
)

# -------------------------
# CASE: RAG flow (PDF upload)
# -------------------------
if option == "Ask using a travel document":
    uploaded_pdf = st.file_uploader("📄 Upload your travel PDF", type=["pdf"])
    if uploaded_pdf:
        # Save uploaded file temporarily
        pdf_path = os.path.join(".", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success(f"✅ Uploaded: {uploaded_pdf.name}")

        # Extract text
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for p in pdf.pages:
                    text += p.extract_text() or ""
            if not text.strip():
                st.warning("No text extracted from PDF; maybe it's scanned images. Consider OCR.")
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            st.stop()

        # Split -> embed -> persist
        chunks = split_text_with_meta(text)
        try:
            with st.spinner("Embedding document and building vectorstore..."):
                vectorstore = build_or_load_vectorstore_from_chunks(chunks)
        except Exception as e:
            st.error(f"❌ Embedding / vectorstore error: {e}")
            st.stop()

        # Query UI
        st.subheader("💬 Ask something about your document")
        user_query = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if not user_query.strip():
                st.warning("Please type a question.")
            else:
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.get_relevant_documents(user_query)
                    if not relevant_docs:
                        st.info("📭 No relevant chunks found in PDF. Falling back to web search.")
                        web_results = web_search_tool(user_query)
                        llm = create_chat_llm(temperature=0.5)
                        prompt = f"""
You are a travel assistant. Summarize the following search results into a clear, friendly, and useful answer.
Search Results:
{web_results}

Question: {user_query}

Answer (be concise and helpful):
"""
                        response = llm.invoke(prompt)
                        final = response.content if hasattr(response, "content") else str(response)
                        st.markdown("### 🌐 Web Search Answer")
                        st.write(final)
                    else:
                        # Build context + request model to answer with citations to chunk index
                        context = "\n\n---\n\n".join([f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(relevant_docs)])
                        prompt = f"""
You are a travel assistant. Use ONLY the provided context. For any fact you state, append a short citation like (source: Chunk 2).
If the answer cannot be found in the context, say "I cannot answer this based on the provided context."

Context:
{context}

Question: {user_query}

Answer:
"""
                        llm = create_chat_llm(temperature=0.3)
                        response = llm.invoke(prompt)
                        final = response.content if hasattr(response, "content") else str(response)
                        st.markdown("### 🧭 Answer from PDF (with provenance)")
                        st.write(final)

                        with st.expander("📘 View retrieved document chunks"):
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.write(doc.page_content)

                except Exception as e:
                    logger.exception("Retrieval / generation failed: %s", e)
                    st.error(f"Error during retrieval/generation: {e}")

# -------------------------
# CASE: Web search only
# -------------------------
elif option == "Search travel info on web":
    st.subheader("🌐 Ask anything travel-related")
    user_query = st.text_input("Where would you like to go or what do you want to know?")
    if st.button("Search"):
        if not user_query.strip():
            st.warning("Please enter a travel query.")
        else:
            try:
                st.info("🔍 Searching the web...")
                web_results = web_search_tool(user_query)
                llm = create_chat_llm(temperature=0.5)
                prompt = f"""
You are a travel assistant. Summarize the following search results into a concise, friendly, and informative answer for a traveler.

Search Results:
{web_results}

Question: {user_query}

Answer:
"""
                response = llm.invoke(prompt)
                final = response.content if hasattr(response, "content") else str(response)
                st.markdown("### 🧭 Travel Insights:")
                st.write(final)
            except Exception as e:
                logger.exception("Web search error: %s", e)
                st.error(f"❌ Web search error: {e}")

# -------------------------
# CASE: Weather only
# -------------------------
elif option == "Check weather in a city":
    city = st.text_input("Enter city name:")
    if st.button("Get Weather"):
        if not city.strip():
            st.warning("Please enter a city name.")
        else:
            try:
                result = weather_tool(city)
                if result.lower().startswith("city not found"):
                    st.error(result)
                else:
                    st.markdown(f"### 🌤 Weather in **{city.title()}**")
                    st.write(result)
            except Exception as e:
                logger.exception("Weather fetch failed: %s", e)
                st.error(f"❌ Error fetching weather: {e}")

# -------------------------
# CASE: Agent (combined tools)
# -------------------------
elif option == "Agent (combined tools)":
    st.subheader("🤖 Agent (uses web search + weather tool)")
    agent_query = st.text_input("Ask the agent anything (e.g., 'Weather in Tokyo and top attractions')")
    if st.button("Run Agent"):
        if not agent_query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                st.info("⚙️ Running agent...")
                # Build a lightweight agent that knows about the web and weather tools.
                llm = create_chat_llm(temperature=0.3)
                agent = initialize_agent([web_tool, weather_tool_wrapped], llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=False)
                # Run
                result = agent.run(agent_query)
                st.markdown("### 🤖 Agent Result")
                st.write(result)
            except Exception as e:
                logger.exception("Agent run failed: %s", e)
                st.error(f"Agent error: {e}")