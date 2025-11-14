# app.py
import os
import time
import logging
import sqlite3
import json
from typing import List, Optional, Tuple
from datetime import datetime, timedelta

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
st.title("🌍 Travel Assistant (RAG + Web Search + Weather + Flights (Amadeus) + DB)")
st.write("Gemini + LangChain + OpenWeather + Amadeus (Flight Offers Search)")

# -------------------------
# Load secrets from Streamlit
# -------------------------
required_secrets = ["GOOGLE_API_KEY", "OPENWEATHER_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]
for s in required_secrets:
    if s not in st.secrets:
        st.error(f"❌ Please configure {s} in Streamlit Secrets.")
        st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENWEATHER_API_KEY = st.secrets["OPENWEATHER_API_KEY"]
AMADEUS_CLIENT_ID = st.secrets["AMADEUS_CLIENT_ID"]
AMADEUS_CLIENT_SECRET = st.secrets["AMADEUS_CLIENT_SECRET"]

# Keep env variables for libs that expect them
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# Constants / Persistence
# -------------------------
PERSIST_DIR = "./chroma_db"  # ensure write access in deployment
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
# Keep your preferred model name here. The create_chat_llm function will probe fallbacks automatically.
CHAT_MODEL_NAME = "gemini-2.0"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# SQLite DB path (persisted alongside app)
DB_PATH = "searches.db"

# Amadeus endpoints (test environment)
AMADEUS_OAUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

# Basic in-memory token cache (module-level)
_amadeus_token_cache = {"access_token": None, "expires_at": 0}

# -------------------------
# Utility: HTTP requests with retries
# -------------------------
def requests_get_with_retries(url, params=None, headers=None, retries=3, backoff=1.0, timeout=8):
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

def requests_post_with_retries(url, data=None, json_body=None, headers=None, retries=3, backoff=1.0, timeout=10):
    for i in range(retries):
        try:
            if json_body is not None:
                resp = requests.post(url, json=json_body, headers=headers, timeout=timeout)
            else:
                resp = requests.post(url, data=data, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            logger.warning("POST error (attempt %d/%d): %s", i + 1, retries, e)
            if i == retries - 1:
                raise
            time.sleep(backoff * (2 ** i))

# -------------------------
# Amadeus: token management
# -------------------------
def get_amadeus_token(force_refresh: bool = False) -> str:
    now = int(time.time())
    token = _amadeus_token_cache.get("access_token")
    expires_at = _amadeus_token_cache.get("expires_at", 0)

    if token and not force_refresh and now < expires_at - 10:
        return token

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET
    }
    try:
        resp = requests_post_with_retries(AMADEUS_OAUTH_URL, data=data, headers=headers, timeout=10)
        js = resp.json()
        access_token = js.get("access_token")
        expires_in = js.get("expires_in", 0)  # seconds
        if not access_token:
            logger.error("Amadeus token response: %s", js)
            raise RuntimeError("Failed to obtain Amadeus access token.")
        _amadeus_token_cache["access_token"] = access_token
        _amadeus_token_cache["expires_at"] = int(time.time()) + int(expires_in)
        logger.info("Obtained Amadeus token, expires in %s sec", expires_in)
        return access_token
    except Exception as e:
        logger.exception("Failed to get Amadeus token: %s", e)
        raise

# -------------------------
# Amadeus: flight search helper
# -------------------------
def amadeus_flight_search(origin: str, destination: str, date_from: str, date_to: Optional[str] = None, adults: int = 1, max_results: int = 5) -> List[dict]:
    token = get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}

    offers = []
    if date_to:
        d1 = datetime.fromisoformat(date_from).date()
        d2 = datetime.fromisoformat(date_to).date()
        days = (d2 - d1).days + 1
        days = min(max(days, 1), 5)
        dates = [(d1 + timedelta(days=i)).isoformat() for i in range(days)]
    else:
        dates = [date_from]

    for dt in dates:
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": destination,
            "departureDate": dt,
            "adults": str(adults),
            "max": str(max_results)
        }
        try:
            resp = requests_get_with_retries(AMADEUS_FLIGHT_OFFERS_URL, params=params, headers=headers, timeout=12)
            data = resp.json()
            day_offers = data.get("data", [])
            for o in day_offers:
                o["_search_date"] = dt
            offers.extend(day_offers)
            if len(offers) >= max_results:
                break
        except requests.HTTPError as he:
            status = getattr(he.response, "status_code", None)
            if status in (401, 403):
                logger.info("Amadeus token maybe expired; refreshing token and retrying once.")
                token = get_amadeus_token(force_refresh=True)
                headers["Authorization"] = f"Bearer {token}"
                try:
                    resp = requests_get_with_retries(AMADEUS_FLIGHT_OFFERS_URL, params=params, headers=headers, timeout=12)
                    data = resp.json()
                    day_offers = data.get("data", [])
                    for o in day_offers:
                        o["_search_date"] = dt
                    offers.extend(day_offers)
                except Exception as e2:
                    logger.exception("Retry after token refresh failed: %s", e2)
            else:
                logger.exception("Amadeus flight offers HTTP error: %s", he)
        except Exception as e:
            logger.exception("Amadeus flight search failed: %s", e)

    return offers[:max_results]

# -------------------------
# Tool: Weather
# -------------------------
def weather_tool(city: str) -> str:
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
    try:
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        return results if isinstance(results, str) else str(results)
    except Exception as e:
        logger.exception("Web search failed: %s", e)
        return f"Web search error: {e}"

# -------------------------
# Tool: Flight Search wrapper for agent
# -------------------------
def flight_search_tool(input_str: str) -> str:
    """Search for flights. Input format: 'ORIGIN,DEST,DATE_FROM,DATE_TO,ADULTS' or 'ORIGIN,DEST,DATE,ADULTS'"""
    try:
        parts = [p.strip() for p in input_str.split(",")]
        if len(parts) == 4:
            origin, destination, date_from, adults = parts
            date_to = None
        elif len(parts) == 5:
            origin, destination, date_from, date_to, adults = parts
        else:
            return "Invalid format. Use: ORIGIN,DEST,DATE_FROM,DATE_TO,ADULTS or ORIGIN,DEST,DATE,ADULTS"

        offers = amadeus_flight_search(
            origin=origin,
            destination=destination,
            date_from=date_from,
            date_to=date_to,
            adults=int(adults),
            max_results=3
        )

        if not offers:
            return f"No flights found from {origin} to {destination}"

        results = []
        for i, o in enumerate(offers[:3], 1):
            price = (o.get("price") or {}).get("grandTotal", "N/A")
            date = o.get("_search_date", "N/A")
            results.append(f"Flight {i}: {origin}→{destination} on {date}, Price: {price}")

        return "\n".join(results)
    except Exception as e:
        logger.exception("Flight search tool error: %s", e)
        return f"Flight search error: {e}"

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

flight_tool = Tool(
    name="flight_search",
    func=flight_search_tool,
    description="Search for flights. Input format: 'ORIGIN,DEST,DATE_FROM,DATE_TO,ADULTS' or 'ORIGIN,DEST,DATE,ADULTS'"
)

# -------------------------
# Helper: create/load persistent Chroma vectorstore
# -------------------------
def build_or_load_vectorstore_from_chunks(chunks: List[str]):
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            try:
                logger.info("Loading existing Chroma vectorstore from %s", PERSIST_DIR)
                vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
                # test call if we have a chunk
                if chunks:
                    try:
                        vectorstore.similarity_search(chunks[0][:50], k=1)
                    except Exception:
                        pass
                return vectorstore
            except Exception as load_error:
                logger.warning("Failed to load existing ChromaDB, creating new: %s", load_error)

        logger.info("Creating new Chroma vectorstore at %s", PERSIST_DIR)
        vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=PERSIST_DIR)
        try:
            vectorstore.persist()
        except Exception as e:
            logger.warning("Chroma persist() failed: %s", e)
        return vectorstore

    except Exception as e:
        logger.exception("Failed to build/load vectorstore: %s", e)
        # Fallback: in-memory Chroma (non-persistent)
        st.warning("Using in-memory vectorstore (changes won't persist).")
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
        return Chroma.from_texts(texts=chunks, embedding=embedding_model)

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
# LangChain LLM helper (Gemini chat wrapper) with robust fallback
# -------------------------
def create_chat_llm(temperature=0.3, show_probe: bool = True) -> ChatGoogleGenerativeAI:
    candidates = []
    if CHAT_MODEL_NAME:
        candidates.append(CHAT_MODEL_NAME)
    # prefer models you see in ListModels; fallback set:
    candidates.extend(["gemini-2.0", "gemini-1.5", "chat-bison-001", "gemini-alpha"])

    attempts = []
    for model_name in candidates:
        if not model_name:
            continue
        try:
            llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
            msg = f"Initialized LLM with model: {model_name}"
            logger.info(msg)
            if show_probe:
                st.success(msg)
            return llm
        except Exception as e:
            err_text = f"Model {model_name} failed: {type(e).__name__}: {e}"
            logger.warning(err_text)
            attempts.append((model_name, str(e)))
            if show_probe:
                st.warning(err_text)

    details = "\n".join([f"- {m}: {err}" for m, err in attempts])
    err_msg = (
        "No Gemini model could be initialized. Tried the following models (errors shown):\n"
        + details
        + "\n\nPossible causes: access level, API version mismatch, or credentials."
    )
    logger.error(err_msg)
    st.error(err_msg)
    raise RuntimeError(err_msg)

# -------------------------
# SQLite DB helpers
# -------------------------
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        return None

def init_db():
    conn = get_db_connection()
    if conn is None:
        return None
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS searches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        query_text TEXT,
        mode TEXT,
        timestamp TEXT,
        result_snippet TEXT
    );
    """)
    conn.commit()
    return conn

def save_search(conn, query_text: str, mode: str, result_snippet: str):
    if conn is None:
        return
    cur = conn.cursor()
    cur.execute("INSERT INTO searches (query_text, mode, timestamp, result_snippet) VALUES (?, ?, ?, ?)",
                (query_text, mode, datetime.utcnow().isoformat(), (result_snippet or "")[:1000]))
    conn.commit()

def safe_save_search(query_text: str, mode: str, result_snippet: str):
    try:
        save_search(conn, query_text, mode, result_snippet)
    except Exception as e:
        logger.warning("Failed to save search to DB: %s", e)

# Initialize DB connection
conn = init_db()
if conn is None:
    st.warning("Database connection failed - search history will not be saved.")

# -------------------------
# Itinerary generator (basic)
# -------------------------
def generate_basic_itinerary(destination: str, days: int = 3) -> str:
    dest = destination.strip()
    if not dest:
        return "Please provide a destination."
    search_text = f"Top attractions in {dest}"
    web_results = web_search_tool(search_text)
    weather_summary = weather_tool(dest)
    plan_lines = [f"**Itinerary for {dest.title()}** — {days} day(s)", f"Weather (current): {weather_summary}", ""]
    attractions = []
    if isinstance(web_results, str):
        candidates = [line.strip() for line in web_results.splitlines() if line.strip()]
        for c in candidates:
            if len(attractions) >= 8:
                break
            snippet = c if len(c) < 150 else c[:147] + "..."
            attractions.append(snippet)
    if not attractions:
        attractions = [f"Explore the city center of {dest.title()}", "Visit the local museum", "Try local cuisine at top restaurants"]
    idx = 0
    for d in range(1, days + 1):
        plan_lines.append(f"### Day {d}")
        for _ in range(2):
            if idx < len(attractions):
                plan_lines.append(f"- {attractions[idx]}")
                idx += 1
            else:
                plan_lines.append("- Free time / explore locally")
        plan_lines.append("")
    plan_lines.append("**Notes:** This is a simple autogenerated plan. Check opening hours, transit, and buy tickets in advance.")
    return "\n".join(plan_lines)

# -------------------------
# Streamlit: Choose mode
# -------------------------
option = st.radio(
    "How can I assist you today?",
    (
        "Ask using a travel document",
        "Search travel info on web",
        "Check weather in a city",
        "Agent (combined tools)",
        "Flight search (Amadeus)",
        "Generate itinerary",
        "View saved searches (DB)"
    )
)

# -------------------------
# CASE: RAG flow (PDF upload)
# -------------------------
if option == "Ask using a travel document":
    st.subheader("📄 Ask using a travel document (PDF)")
    uploaded_pdf = st.file_uploader("📄 Upload your travel PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_path = os.path.join(".", uploaded_pdf.name)
        try:
            # Save uploaded file
            with open(pdf_path, "wb") as f:
                f.write(uploaded_pdf.read())
            st.success(f"✅ Uploaded: {uploaded_pdf.name}")

            # Extract text
            text = ""
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for p in pdf.pages:
                        try:
                            text += p.extract_text() or ""
                        except Exception as page_err:
                            logger.warning("Error extracting text from a page: %s", page_err)
                            continue
                if not text.strip():
                    st.warning("No text extracted from PDF; maybe it's scanned images. Consider OCR.")
            except Exception as e:
                st.error(f"❌ Failed to read PDF: {e}")
                raise  # go to outer except to allow cleanup

            # Split -> embed -> persist (vectorstore)
            chunks = split_text_with_meta(text)
            try:
                with st.spinner("Embedding document and building vectorstore..."):
                    vectorstore = build_or_load_vectorstore_from_chunks(chunks)
            except Exception as e:
                st.error(f"❌ Embedding / vectorstore error: {e}")
                raise  # go to outer except to allow cleanup

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
                            safe_save_search(user_query, "rag_fallback_web", final)
                        else:
                            context = "\n\n---\n\n".join(
                                [f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(relevant_docs)]
                            )
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
                            safe_save_search(user_query, "rag_doc", final)

                            with st.expander("📘 View retrieved document chunks"):
                                for i, doc in enumerate(relevant_docs):
                                    st.markdown(f"**Chunk {i+1}:**")
                                    st.write(doc.page_content)

                    except Exception as q_err:
                        logger.exception("RAG query failed: %s", q_err)
                        st.error(f"❌ Query processing error: {q_err}")

        except Exception as outer_err:
            # General cleanup and error reporting
            logger.exception("PDF processing failed: %s", outer_err)
            st.error(f"❌ PDF processing error: {outer_err}")
        finally:
            # Clean up the temporary uploaded PDF file (if present)
            try:
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)
            except Exception as cleanup_err:
                logger.warning("Failed to remove temporary PDF %s: %s", pdf_path, cleanup_err)


# -------------------------
# CASE: Weather only
# -------------------------
elif option == "Check weather in a city":
    st.subheader("🌤 Check weather in a city")

    city = st.text_input("Enter city name:", key="weather_city")
    if st.button("Get Weather", key="get_weather_btn"):
        if not city or not city.strip():
            st.warning("Please enter a city name.")
        else:
            try:
                result = weather_tool(city.strip())
                if isinstance(result, str) and result.lower().startswith("city not found"):
                    st.error(result)
                else:
                    st.markdown(f"### 🌤 Weather in **{city.title()}**")
                    st.write(result)
                    safe_save_search(city, "weather", result)
            except Exception as e:
                logger.exception("Weather fetch failed: %s", e)
                st.error(f"❌ Error fetching weather: {e}")

# -------------------------
# CASE: Agent (combined tools)
# -------------------------
elif option == "Agent (combined tools)":
    st.subheader("🤖 Agent (uses web search + weather + flight tools)")
    agent_query = st.text_input("Ask the agent anything (e.g., 'Weather in Tokyo and top attractions')")
    if st.button("Run Agent"):
        if not agent_query.strip():
            st.warning("Please enter a query.")
        else:
            try:
                st.info("⚙️ Running agent...")
                llm = create_chat_llm(temperature=0.3)
                agent = initialize_agent(
                    [web_tool, weather_tool_wrapped, flight_tool],
                    llm,
                    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                    verbose=False
                )
                result = agent.run(agent_query)
                st.markdown("### 🤖 Agent Result")
                st.write(result)
                safe_save_search(agent_query, "agent", result)
            except Exception as e:
                logger.exception("Agent run failed: %s", e)
                st.error(f"Agent error: {e}")

# -------------------------
# CASE: Flight search (Amadeus)
# -------------------------
elif option == "Flight search (Amadeus)":
    st.subheader("✈️ Flight search (Amadeus Flight Offers)")
    st.markdown("Enter origin/destination IATA or city code and a date or date range. This uses Amadeus Self-Service test environment.")
    col1, col2 = st.columns(2)
    with col1:
        origin = st.text_input("From (IATA or city, e.g., DEL or New Delhi)", value="DEL")
    with col2:
        dest = st.text_input("To (IATA or city, e.g., BLR or Bengaluru)", value="BLR")
    c1, c2 = st.columns(2)
    with c1:
        date_from = st.date_input("Earliest departure", value=datetime.utcnow().date())
    with c2:
        date_to = st.date_input("Latest departure", value=(datetime.utcnow().date() + timedelta(days=3)))
    adults = st.number_input("Adults", min_value=1, value=1)
    max_results = st.number_input("Max results", min_value=1, value=5)
    if st.button("Search Flights (Amadeus)"):
        try:
            st.info("Searching Amadeus (may use multiple dates within the range)...")
            offers = amadeus_flight_search(
                origin=origin.strip(),
                destination=dest.strip(),
                date_from=date_from.isoformat(),
                date_to=date_to.isoformat() if date_to else None,
                adults=int(adults),
                max_results=int(max_results)
            )
            if not offers:
                st.info("No offers found for the given parameters.")
                safe_save_search(f"flights: {origin}->{dest} ({date_from}..{date_to})", "amadeus_flights", "no results")
            else:
                for i, o in enumerate(offers):
                    try:
                        # Defensive: ensure dict
                        if not isinstance(o, dict):
                            st.write(f"Offer {i+1}: (type {type(o).__name__})")
                            st.code(str(o)[:800])
                            continue

                        price = None
                        price_info = o.get("price") or {}
                        if isinstance(price_info, dict):
                            price = price_info.get("grandTotal") or price_info.get("total") or price_info.get("totalPrice")
                        # fallback from offerItems
                        if not price:
                            offer_items = o.get("offerItems") or []
                            if isinstance(offer_items, list) and offer_items:
                                first = offer_items[0]
                                price = (first.get("price") or {}).get("total") if isinstance(first, dict) else None

                        dep_date = o.get("_search_date", "N/A")
                        price_text = price if price else "N/A"
                        st.markdown(f"**Offer {i+1}** — Date: {dep_date} — Price: {price_text}")

                        # Itineraries / segments tolerant handling
                        itineraries = []
                        if o.get("itineraries"):
                            itineraries = o.get("itineraries")
                        elif o.get("offerItems"):
                            for oi in o.get("offerItems", []):
                                services = oi.get("services", [])
                                for svc in services:
                                    segments = svc.get("segments", [])
                                    if segments:
                                        itineraries.append({"segments": segments})

                        if isinstance(itineraries, dict):
                            itineraries = [itineraries]

                        if itineraries:
                            st.write("Itineraries:")
                            for itin in itineraries:
                                segments = itin.get("segments", []) or []
                                for seg in segments[:6]:
                                    dep = seg.get("departure") if isinstance(seg, dict) else {}
                                    arr = seg.get("arrival") if isinstance(seg, dict) else {}
                                    dep_code = dep.get("iataCode") or dep.get("iata") or "?"
                                    dep_time = dep.get("at") or dep.get("time") or "?"
                                    arr_code = arr.get("iataCode") or arr.get("iata") or "?"
                                    arr_time = arr.get("at") or arr.get("time") or "?"
                                    st.write(f"- {dep_code} {dep_time} → {arr_code} {arr_time}")

                        st.write("Raw (truncated):")
                        snippet = json.dumps(o, default=str)[:800]
                        st.code(snippet)
                        st.write("---")
                    except Exception as e_inner:
                        st.error(f"Error rendering offer {i+1}: {e_inner}")
                        logger.exception("Error rendering offer %d: %s", i+1, e_inner)

                safe_save_search(f"flights: {origin}->{dest}", "amadeus_flights", json.dumps(offers[0], default=str))

        except Exception as e:
            logger.exception("Amadeus flight search error: %s", e)
            st.error(f"Flight search error: {e}")

# -------------------------
# CASE: Generate itinerary
# -------------------------
elif option == "Generate itinerary":
    st.subheader("🗺️ Basic Itinerary Generator")
    dest = st.text_input("Destination (city):", value="Rome")
    days = st.number_input("Days", value=3, min_value=1, max_value=14)
    if st.button("Create itinerary"):
        try:
            it = generate_basic_itinerary(dest, int(days))
            st.markdown(it)
            safe_save_search(f"itinerary: {dest} x {days}", "itinerary", it)
        except Exception as e:
            logger.exception("Itinerary failed: %s", e)
            st.error(f"❌ Itinerary error: {e}")

# -------------------------
# CASE: View saved searches (DB)
# -------------------------
elif option == "View saved searches (DB)":
    st.subheader("📚 Saved searches (SQLite)")
    if conn is None:
        st.error("Database not available")
    else:
        cur = conn.cursor()
        cur.execute("SELECT id, query_text, mode, timestamp FROM searches ORDER BY id DESC LIMIT 100")
        rows = cur.fetchall()
        if not rows:
            st.info("No saved searches yet.")
        else:
            for r in rows:
                sid, q, mode, ts = r
                with st.expander(f"[{sid}] {mode} • {q} • {ts}"):
                    cur.execute("SELECT result_snippet FROM searches WHERE id=?", (sid,))
                    snippet = cur.fetchone()
                    st.write(snippet[0] if snippet else "")
                    if st.button(f"Delete {sid}", key=f"del_{sid}"):
                        cur.execute("DELETE FROM searches WHERE id=?", (sid,))
                        conn.commit()
                        st.experimental_rerun()

# End of file