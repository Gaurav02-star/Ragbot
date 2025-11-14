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
# Use the available models from your API
CHAT_MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",  # Primary - this should work
    "models/gemini-2.5-flash-preview-05-20",  # Fallback
    "models/gemini-2.5-pro-preview-03-25",  # Fallback
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
    
    # If all models fail
    st.error(f"❌ All model attempts failed. Last error: {last_error}")
    raise RuntimeError(f"No working model found. Last error: {last_error}")

# -------------------------
# Simple Text Search (Fallback when embeddings fail)
# -------------------------
def simple_text_search(text: str, query: str, top_k: int = 3) -> List[str]:
    """
    Simple keyword-based search as fallback when embeddings are not available.
    Returns the most relevant chunks based on keyword matching.
    """
    # Split text into sentences or small chunks
    sentences = []
    for paragraph in text.split('\n'):
        for sentence in paragraph.split('.'):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                sentences.append(sentence)
    
    # Score sentences based on query keyword matches
    query_words = query.lower().split()
    scored_sentences = []
    
    for sentence in sentences:
        score = 0
        sentence_lower = sentence.lower()
        for word in query_words:
            if len(word) > 3 and word in sentence_lower:
                score += 1
        if score > 0:
            scored_sentences.append((score, sentence))
    
    # Sort by score and return top results
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    return [sentence for _, sentence in scored_sentences[:top_k]]

# -------------------------
# Embedding with Quota Handling
# -------------------------
def create_embedding_model():
    """Create embedding model with quota error handling"""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        # Test the embedding model with a small query
        test_embedding = embedding_model.embed_query("test")
        return embedding_model
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            st.warning("⚠️ Embedding API quota exceeded. Using keyword-based search instead.")
            return None
        else:
            raise e

def build_or_load_vectorstore_from_chunks(chunks: List[str]):
    """Create or load a persisted Chroma vectorstore with quota handling"""
    try:
        embedding_model = create_embedding_model()
        
        if embedding_model is None:
            # Embedding not available, return None to use fallback
            return None
            
        # If persisted data exists, try to open
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
        if "quota" in str(e).lower() or "429" in str(e):
            st.warning("⚠️ Embedding API quota exceeded. Using keyword-based search instead.")
            return None
        else:
            logger.exception("Failed to build/load vectorstore: %s", e)
            raise

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
# CASE: Web search only
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
                
                # Use Gemini for summarization
                try:
                    llm = create_chat_llm(temperature=0.5)
                    prompt = f"""
You are a helpful travel assistant. Please analyze the following search results and provide a comprehensive, well-organized answer to the user's question.

SEARCH RESULTS:
{web_results}

USER'S QUESTION: {user_query}

Please provide a detailed answer with these sections if applicable:
1. Top attractions/places to visit
2. Best time to visit
3. Travel tips
4. Local highlights

Make it engaging and practical for travelers:"""
                    response = llm.invoke(prompt)
                    final = response.content if hasattr(response, "content") else str(response)
                    st.markdown("### 🧭 Travel Insights")
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
# CASE: RAG flow (PDF upload) - WITH QUOTA HANDLING
# -------------------------
elif option == "Ask using a travel document":
    st.subheader("📄 Document-based Travel Assistant")
    
    # Show quota warning
    st.info("💡 **Note:** If you see quota errors, the app will automatically use keyword search instead of AI embeddings.")
    
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

        # Store the extracted text for fallback search
        st.session_state['pdf_text'] = text
        
        chunks = split_text_with_meta(text)
        
        user_query = st.text_input("Enter your question here:", "where to visit in summer in india")
        if st.button("Get Answer"):
            if not user_query.strip():
                st.warning("Please type a question.")
            else:
                try:
                    with st.spinner("Searching document..."):
                        vectorstore = build_or_load_vectorstore_from_chunks(chunks)
                        
                        if vectorstore is None:
                            # Use keyword-based fallback search
                            st.info("🔍 Using keyword search (AI embeddings unavailable)")
                            relevant_chunks = simple_text_search(text, user_query, top_k=5)
                            
                            if not relevant_chunks:
                                st.info("📭 No relevant content found in PDF. Try web search instead.")
                                web_results = web_search_tool(user_query)
                                st.markdown("### 🌐 Web Search Results")
                                st.write(web_results)
                            else:
                                context = "\n\n---\n\n".join([f"[Section {i+1}]: {chunk}" for i, chunk in enumerate(relevant_chunks)])
                                
                                try:
                                    llm = create_chat_llm(temperature=0.3)
                                    prompt = f"""Based on the following document sections, answer the question:

DOCUMENT SECTIONS:
{context}

QUESTION: {user_query}

Please provide a helpful answer based on the document content:"""
                                    
                                    response = llm.invoke(prompt)
                                    final = response.content if hasattr(response, "content") else str(response)
                                    st.markdown("### 📖 Answer from Document")
                                    st.write(final)
                                    
                                    with st.expander("View relevant document sections"):
                                        for i, chunk in enumerate(relevant_chunks):
                                            st.markdown(f"**Section {i+1}:**")
                                            st.write(chunk)
                                            
                                except Exception as e:
                                    st.warning(f"AI processing failed, showing relevant sections: {e}")
                                    st.markdown("### 📚 Relevant Document Sections")
                                    for i, chunk in enumerate(relevant_chunks):
                                        st.markdown(f"**Section {i+1}:**")
                                        st.write(chunk)
                        else:
                            # Use vectorstore for semantic search
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

Provide a detailed and helpful answer:"""
                                    response = llm.invoke(prompt)
                                    final = response.content if hasattr(response, "content") else str(response)
                                    st.markdown("### 🧭 Answer from PDF")
                                    st.write(final)
                                    
                                    with st.expander("View retrieved document sections"):
                                        for i, doc in enumerate(relevant_docs):
                                            st.markdown(f"**Section {i+1}:**")
                                            st.write(doc.page_content)
                                            
                                except Exception as e:
                                    st.warning(f"LLM failed, showing raw context: {e}")
                                    st.markdown("### 📚 Relevant Document Sections")
                                    for i, doc in enumerate(relevant_docs):
                                        st.markdown(f"**Section {i+1}:**")
                                        st.write(doc.page_content)

                except Exception as e:
                    logger.exception("Document search failed: %s", e)
                    if "quota" in str(e).lower():
                        st.error("❌ Embedding quota exceeded. Please try the web search option instead, or upload a smaller document.")
                    else:
                        st.error(f"Error during document search: {e}")

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
                
                # Extract city for weather if mentioned
                city = "Goa"  # default
                if "weather" in agent_query.lower():
                    # Simple city extraction - you could make this smarter
                    for word in agent_query.split():
                        if word.lower() not in ['weather', 'in', 'and', 'the']:
                            city = word
                            break
                
                # Get weather if relevant
                weather_info = ""
                if "weather" in agent_query.lower():
                    weather_info = weather_tool(city)
                
                # Get web search results
                search_query = agent_query
                if "weather" in agent_query.lower():
                    # Remove weather part for search
                    search_query = agent_query.replace("weather", "").replace("Weather", "").strip()
                
                search_results = web_search_tool(search_query)
                
                # Combine and process with Gemini
                try:
                    llm = create_chat_llm(temperature=0.3)
                    
                    combined_context = f"WEB SEARCH RESULTS:\n{search_results}"
                    if weather_info:
                        combined_context += f"\n\nWEATHER INFORMATION:\n{weather_info}"
                    
                    prompt = f"""As a knowledgeable travel assistant, provide comprehensive information based on the following data:

{combined_context}

USER'S QUESTION: {agent_query}

Please provide a well-structured answer that includes:
- Key attractions and activities
- Practical travel information
- Current conditions (if weather data available)
- Helpful tips for visitors

Make it engaging and useful for someone planning a trip:"""
                    
                    response = llm.invoke(prompt)
                    result = response.content if hasattr(response, "content") else str(response)
                    st.markdown("### 🤖 Travel Assistance")
                    st.write(result)
                    
                    # Show raw data in expander
                    with st.expander("View raw data sources"):
                        if weather_info:
                            st.markdown("**🌤 Weather Data:**")
                            st.write(weather_info)
                        st.markdown("**🔍 Web Search Results:**")
                        st.write(search_results[:1000] + "..." if len(search_results) > 1000 else search_results)
                        
                except Exception as e:
                    st.warning(f"Gemini processing failed, showing raw results: {e}")
                    st.markdown("### 🔍 Raw Results")
                    if weather_info:
                        st.markdown("**🌤 Weather:**")
                        st.write(weather_info)
                    st.markdown("**🔍 Web Search:**")
                    st.write(search_results)
                    
            except Exception as e:
                logger.exception("Assistant failed: %s", e)
                st.error(f"Assistant error: {e}")

# Add info about quota limits
with st.sidebar:
    st.markdown("---")
    st.info("""
    **API Usage Notes:**
    - Chat models: ✅ Working
    - Embeddings: ⚠️ Limited quota
    - Web search: ✅ Unlimited
    - Weather: ✅ Working
    """)