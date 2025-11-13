# app.py
import os
import time
import logging
import sqlite3
import json
from typing import List, Optional
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
    """
    Obtain an OAuth token from Amadeus (client credentials). Cache until near expiry.
    """
    now = int(time.time())
    token = _amadeus_token_cache.get("access_token")
    expires_at = _amadeus_token_cache.get("expires_at", 0)

    if token and not force_refresh and now < expires_at - 10:
        return token

    # Request a new token
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
    """
    Search Amadeus Flight Offers. If date_to provided and different, this will iterate across the date range
    (bounded to a small number of days) and collect top offers per day.
    - date_from and date_to are 'YYYY-MM-DD' strings.
    """
    token = get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}

    offers = []
    # if user provided a date range, iterate up to 5 days to limit calls
    if date_to:
        d1 = datetime.fromisoformat(date_from).date()
        d2 = datetime.fromisoformat(date_to).date()
        days = (d2 - d1).days + 1
        days = min(max(days, 1), 5)  # limit to 5 days to avoid excessive calls
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
            # attach date info and raw meta
            for o in day_offers:
                o["_search_date"] = dt
            offers.extend(day_offers)
            # break early if we already have enough offers
            if len(offers) >= max_results:
                break
        except requests.HTTPError as he:
            # If 401/403 (token expired/invalid), try refresh token once
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

    # return at most max_results offers
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
# LangChain LLM helper (Gemini chat wrapper)
# -------------------------
def create_chat_llm(temperature=0.3):
    return ChatGoogleGenerativeAI(model=CHAT_MODEL_NAME, temperature=temperature)

# -------------------------
# SQLite DB helpers
# -------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
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
    cur = conn.cursor()
    cur.execute("INSERT INTO searches (query_text, mode, timestamp, result_snippet) VALUES (?, ?, ?, ?)",
                (query_text, mode, datetime.utcnow().isoformat(), result_snippet[:1000]))
    conn.commit()

# Initialize DB connection
conn = init_db()

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
# RAG / Web / Weather / Agent / Itinerary / DB flows
# (these are similar to your earlier flows, left largely unchanged)
# -------------------------
# (Omitted here for brevity in this code snippet comment block — include the previous RAG/web/weather/agent/itinerary DB cases
# exactly as in your previous app, replacing the Flight search case below)
# For the full file, these blocks are present (they are identical to your previous app)...
# -------------------------
# CASE: Flight search (Amadeus)
# -------------------------
if option == "Flight search (Amadeus)":
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
                save_search(conn, f"flights: {origin}->{dest} ({date_from}..{date_to})", "amadeus_flights", "no results")
            else:
                for i, o in enumerate(offers):
                    price = None
                    if isinstance(o, dict):
                        # Amadeus offers have price info under 'price' or 'offerItems' fields depending on version
                        price = o.get("price", {}).get("grandTotal") or o.get("price", {}).get("total") if o.get("price") else None
                    # fallback snippet building
                    snippet = json.dumps(o, default=str)[:800]
                    dep_date = o.get("_search_date", "N/A")
                    st.markdown(f"**Offer {i+1}** — Date: {dep_date} — Price: {price if price else 'N/A'}")
                    # show a few details if present
                    try:
                        # look into itineraries -> segments for readable times
                        itineraries = o.get("itineraries", [])
                        for itin in itineraries:
                            st.write("Itinerary:")
                            for seg in itin.get("segments", [])[:2]:
                                st.write(f"- {seg.get('departure',{}).get('iataCode','?')} {seg.get('departure',{}).get('at','?')} → {seg.get('arrival',{}).get('iataCode','?')} {seg.get('arrival',{}).get('at','?')}")
                    except Exception:
                        pass
                    st.write("Raw (truncated):")
                    st.code(snippet)
                    st.write("---")
                save_search(conn, f"flights: {origin}->{dest}", "amadeus_flights", json.dumps(offers[0], default=str))
        except Exception as e:
            logger.exception("Amadeus flight search error: %s", e)
            st.error(f"Flight search error: {e}")

# -------------------------
# For complete file: include your other radio branches
# (RAG, Web search, Weather, Agent, Itinerary, View DB)
# The above Flight search branch integrates Amadeus and replaces the Tequila block.
# -------------------------
