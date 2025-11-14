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
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Please configure GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")

# Keep env variables for libraries that expect them
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# Constants / Persistence
# -------------------------
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "models/embedding-001"
# Try these model names in order
CHAT_MODEL_CANDIDATES = [
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest", 
    "gemini-pro",
    "models/gemini-pro",
    "gemini-1.0-pro",
]
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
    if not OPENWEATHER_API_KEY:
        return "OpenWeather API key not configured."
    
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

# -------------------------
# LLM Creation with Model Fallback
# -------------------------
def create_chat_llm(temperature: float = 0.3):
    """Create LLM with fallback for different model names"""
    last_error = None
    
    for model_name in CHAT_MODEL_CANDIDATES:
        try:
            logger.info(f"Trying model: {model_name}")
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=GOOGLE_API_KEY
            )
            # Test with a simple prompt to verify the model works
            test_response = llm.invoke("Say 'Hello'")
            if test_response:
                logger.info(f"✅ Successfully loaded model: {model_name}")
                return llm
        except Exception as e:
            last_error = e
            logger.warning(f"❌ Model {model_name} failed: {e}")
            continue
    
    # If all models fail, show available models
    st.error(f"❌ All model attempts failed. Last error: {last_error}")
    st.info("🔍 Checking available models...")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        models = genai.list_models()
        available_models = []
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                available_models.append(model.name)
        
        if available_models:
            st.write("Available models for your API key:")
            for model in available_models:
                st.write(f"- {model}")
        else:
            st.write("No generateContent models found.")
    except Exception as e:
        st.write(f"Could not list models: {e}")
    
    raise RuntimeError(f"No working model found. Last error: {last_error}")

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
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            logger.info("Loading existing Chroma vectorstore from %s", PERSIST_DIR)
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        else:
            logger.info("Creating new Chroma vectorstore at %s", PERSIST_DIR)
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=PERSIST_DIR)
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
# Streamlit: Choose mode
# -------------------------
option = st.radio(
    "How can I assist you today?",
    ("Search travel info on web", "Check weather in a city", "Ask using a travel document", "Agent (combined tools)")
)

# -------------------------
# CASE: Web search only (Simplified - Direct DuckDuckGo results)
# -------------------------
if option == "Search travel info on web":
    st.subheader("🌐 Ask anything travel-related")
    user_query = st.text_input("Where would you like to go or what do you want to know?", "where should I visit in goa")
    
    if st.button("Search"):
        if not user_query.strip():
            st.warning("Please enter a travel query.")
        else:
            try:
                st.info("🔍 Searching the web...")
                web_results = web_search_tool(user_query)
                
                # Try to use Gemini for summarization, but fallback to direct results
                try:
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
                    logger.warning(f"Gemini summarization failed, showing raw results: {e}")
                    st.markdown("### 🔍 Search Results (Direct):")
                    st.write(web_results)
                    
            except Exception as e:
                logger.exception("Web search error: %s", e)
                st.error(f"❌ Search error: {e}")

# -------------------------
# CASE: Weather only
# -------------------------
elif option == "Check weather in a city":
    st.subheader("🌤 Check Weather")
    city = st.text_input("Enter city name:", "Goa")
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
# CASE: RAG flow (PDF upload)
# -------------------------
elif option == "Ask using a travel document":
    st.subheader("📄 Document-based Travel Assistant")
    uploaded_pdf = st.file_uploader("Upload your travel PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_path = os.path.join(".", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success(f"✅ Uploaded: {uploaded_pdf.name}")

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

        chunks = split_text_with_meta(text)
        try:
            with st.spinner("Embedding document and building vectorstore..."):
                vectorstore = build_or_load_vectorstore_from_chunks(chunks)
        except Exception as e:
            st.error(f"❌ Embedding / vectorstore error: {e}")
            st.stop()

        user_query = st.text_input("Enter your question here:")
        if st.button("Get Answer"):
            if not user_query.strip():
                st.warning("Please type a question.")
            else:
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.get_relevant_documents(user_query)
                    if not relevant_docs:
                        st.info("📭 No relevant chunks found in PDF.")
                    else:
                        context = "\n\n---\n\n".join([f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(relevant_docs)])
                        try:
                            llm = create_chat_llm(temperature=0.3)
                            prompt = f"""Use this context to answer the question:

Context:
{context}

Question: {user_query}

Answer:"""
                            response = llm.invoke(prompt)
                            final = response.content if hasattr(response, "content") else str(response)
                            st.markdown("### 🧭 Answer from PDF")
                            st.write(final)
                        except Exception as e:
                            st.warning(f"LLM failed, showing raw context: {e}")
                            st.markdown("### 📚 Relevant Document Sections")
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Section {i+1}:**")
                                st.write(doc.page_content)

                except Exception as e:
                    logger.exception("Retrieval failed: %s", e)
                    st.error(f"Error during retrieval: {e}")

# -------------------------
# CASE: Agent (combined tools)
# -------------------------
elif option == "Agent (combined tools)":
    st.subheader("🤖 Smart Assistant (Web + Weather)")
    agent_query = st.text_input("Ask anything (e.g., 'Weather in Tokyo and top attractions')", 
                               "Best places to visit in Goa and current weather")
    if st.button("Run Assistant"):
        if not agent_query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                st.info("⚙️ Processing your request...")
                # For now, use direct tool calls instead of agent
                weather_part = ""
                if "weather" in agent_query.lower():
                    city = "Goa"  # Default, you could extract this
                    weather_part = weather_tool(city)
                
                search_part = web_search_tool(agent_query)
                
                try:
                    llm = create_chat_llm(temperature=0.3)
                    combined_info = f"Web Search Results: {search_part}"
                    if weather_part:
                        combined_info += f"\n\nWeather Information: {weather_part}"
                    
                    prompt = f"""As a travel assistant, provide helpful information based on this data:

{combined_info}

Question: {agent_query}

Provide a comprehensive, friendly answer:"""
                    
                    response = llm.invoke(prompt)
                    result = response.content if hasattr(response, "content") else str(response)
                    st.markdown("### 🤖 Travel Assistance")
                    st.write(result)
                    
                except Exception as e:
                    st.warning(f"Gemini processing failed, showing raw results: {e}")
                    st.markdown("### 🔍 Raw Results")
                    if weather_part:
                        st.markdown("**🌤 Weather:**")
                        st.write(weather_part)
                    st.markdown("**🔍 Web Search:**")
                    st.write(search_part)
                    
            except Exception as e:
                logger.exception("Assistant failed: %s", e)
                st.error(f"Assistant error: {e}")

# Add model debug button
with st.sidebar:
    if st.button("🛠 Debug: Check Available Models"):
        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            models = genai.list_models()
            st.write("### Available Models:")
            for model in models:
                if 'generateContent' in model.supported_generation_methods:
                    st.write(f"✅ **{model.name}**")
                    st.write(f"   Methods: {model.supported_generation_methods}")
        except Exception as e:
            st.error(f"Error checking models: {e}")