# app.py
import os
import time
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any

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
st.title("🌍 Travel Assistant (RAG + Web Search + Weather + Flights)")
st.write("Your AI-powered travel companion — Gemini + LangChain + Amadeus + OpenWeather")

# -------------------------
# Load secrets from Streamlit
# -------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ Please configure GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
OPENWEATHER_API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
AMADEUS_CLIENT_ID = st.secrets.get("AMADEUS_CLIENT_ID", "")
AMADEUS_CLIENT_SECRET = st.secrets.get("AMADEUS_CLIENT_SECRET", "")

# Keep env variables for libraries that expect them
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# -------------------------
# Constants / Persistence
# -------------------------
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "models/embedding-001"
CHAT_MODEL_CANDIDATES = [
    "models/gemini-2.5-flash",
    "models/gemini-2.5-flash-preview-05-20",
    "models/gemini-2.5-pro-preview-03-25",
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DB_PATH = "travel_assistant.db"

# -------------------------
# SQLite Database Setup
# -------------------------
def init_database():
    """Initialize SQLite database for saving searches and itineraries"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create searches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_type TEXT NOT NULL,
            result_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create itineraries table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS itineraries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            destination TEXT NOT NULL,
            duration_days INTEGER,
            itinerary_data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create flight searches table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flight_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            departure_date TEXT NOT NULL,
            return_date TEXT,
            results_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    conn.close()

def save_search(query: str, search_type: str, result_text: str = ""):
    """Save search query and results to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO searches (query, search_type, result_text) VALUES (?, ?, ?)",
        (query, search_type, result_text[:1000])  # Limit result text length
    )
    conn.commit()
    conn.close()

def save_itinerary(title: str, destination: str, duration_days: int, itinerary_data: Dict):
    """Save itinerary to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO itineraries (title, destination, duration_days, itinerary_data) VALUES (?, ?, ?, ?)",
        (title, destination, duration_days, json.dumps(itinerary_data))
    )
    conn.commit()
    conn.close()

def save_flight_search(origin: str, destination: str, departure_date: str, return_date: str = None, results_count: int = 0):
    """Save flight search to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO flight_searches (origin, destination, departure_date, return_date, results_count) VALUES (?, ?, ?, ?, ?)",
        (origin, destination, departure_date, return_date, results_count)
    )
    conn.commit()
    conn.close()

def get_recent_searches(limit: int = 10):
    """Get recent searches from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query, search_type, created_at FROM searches ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return results

def get_saved_itineraries():
    """Get all saved itineraries"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, title, destination, duration_days, created_at FROM itineraries ORDER BY created_at DESC")
    results = cursor.fetchall()
    conn.close()
    return results

# Initialize database
init_database()

# -------------------------
# Amadeus Flight API Integration
# -------------------------
class AmadeusClient:
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.access_token = None
        self.token_expiry = None
    
    def get_access_token(self):
        """Get access token from Amadeus API"""
        if self.access_token and self.token_expiry and datetime.now() < self.token_expiry:
            return self.access_token
            
        try:
            url = "https://test.api.amadeus.com/v1/security/oauth2/token"
            data = {
                'grant_type': 'client_credentials',
                'client_id': self.client_id,
                'client_secret': self.client_secret
            }
            headers = {
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            # Set expiry 5 minutes before actual expiry to be safe
            expires_in = token_data.get('expires_in', 1799) - 300
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get Amadeus access token: {e}")
            return None
    
    def search_flights(self, origin: str, destination: str, departure_date: str, return_date: str = None, adults: int = 1):
        """Search for flights using Amadeus API"""
        try:
            token = self.get_access_token()
            if not token:
                return {"error": "Failed to authenticate with Amadeus API"}
            
            url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
            headers = {
                'Authorization': f'Bearer {token}'
            }
            params = {
                'originLocationCode': origin.upper(),
                'destinationLocationCode': destination.upper(),
                'departureDate': departure_date,
                'adults': adults,
                'max': 10  # Limit results
            }
            
            if return_date:
                params['returnDate'] = return_date
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                flights = self._parse_flight_data(data)
                save_flight_search(origin, destination, departure_date, return_date, len(flights))
                return flights
            else:
                logger.error(f"Amadeus API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Flight search error: {e}")
            return {"error": str(e)}
    
    def _parse_flight_data(self, data: Dict) -> List[Dict]:
        """Parse flight data from Amadeus response"""
        flights = []
        
        if not data.get('data'):
            return flights
        
        for offer in data['data']:
            flight_info = {
                'id': offer['id'],
                'price': offer['price']['total'],
                'currency': offer['price']['currency'],
                'itineraries': []
            }
            
            for itinerary in offer['itineraries']:
                segments = []
                for segment in itinerary['segments']:
                    segment_info = {
                        'departure': {
                            'airport': segment['departure']['iataCode'],
                            'time': segment['departure']['at']
                        },
                        'arrival': {
                            'airport': segment['arrival']['iataCode'],
                            'time': segment['arrival']['at']
                        },
                        'airline': segment['carrierCode'],
                        'flight_number': segment['number'],
                        'duration': segment['duration']
                    }
                    segments.append(segment_info)
                
                flight_info['itineraries'].append({
                    'duration': itinerary['duration'],
                    'segments': segments
                })
            
            flights.append(flight_info)
        
        return flights

# Initialize Amadeus client
amadeus_client = AmadeusClient(AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET)

# -------------------------
# Itinerary Generation
# -------------------------
def generate_itinerary(destination: str, duration_days: int, interests: List[str], budget: str = "medium"):
    """Generate a travel itinerary using AI"""
    try:
        llm = create_chat_llm(temperature=0.7)
        
        prompt = f"""
        Create a detailed {duration_days}-day travel itinerary for {destination}.
        
        Traveler Interests: {', '.join(interests)}
        Budget Level: {budget}
        
        Please structure the itinerary with:
        1. Daily schedule with morning, afternoon, and evening activities
        2. Recommended restaurants/cafes for each day
        3. Transportation tips between locations
        4. Estimated costs where possible
        5. Cultural highlights and must-see attractions
        
        Make it practical, engaging, and tailored to the interests mentioned.
        """
        
        response = llm.invoke(prompt)
        itinerary_text = response.content if hasattr(response, 'content') else str(response)
        
        # Create structured itinerary data
        itinerary_data = {
            'destination': destination,
            'duration_days': duration_days,
            'interests': interests,
            'budget': budget,
            'itinerary_text': itinerary_text,
            'generated_at': datetime.now().isoformat()
        }
        
        # Save to database
        title = f"{duration_days}-Day {destination} Trip"
        save_itinerary(title, destination, duration_days, itinerary_data)
        
        return itinerary_data
        
    except Exception as e:
        logger.error(f"Itinerary generation error: {e}")
        return {"error": str(e)}

# -------------------------
# Existing Utility Functions (from previous code)
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

def web_search_tool(query: str) -> str:
    """Run a DuckDuckGoSearchRun search and return a concise combined string."""
    try:
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        return results if isinstance(results, str) else str(results)
    except Exception as e:
        logger.exception("Web search failed: %s", e)
        return f"Web search error: {e}"

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
    
    st.error(f"❌ All model attempts failed. Last error: {last_error}")
    raise RuntimeError(f"No working model found. Last error: {last_error}")

# ... (Include all the other existing functions like simple_text_search, create_embedding_model, build_or_load_vectorstore_from_chunks, split_text_with_meta)

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
# Streamlit Navigation
# -------------------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("Go to", [
    "Travel Search", 
    "Flight Search", 
    "Itinerary Generator",
    "Saved Data",
    "Document Search"
])

# -------------------------
# PAGE: Travel Search (Existing functionality)
# -------------------------
if page == "Travel Search":
    st.header("🔍 Travel Information Search")
    
    option = st.radio(
        "Search type:",
        ("Web Search", "Weather Check", "Smart Assistant")
    )
    
    if option == "Web Search":
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
                        
                        # Save search to database
                        save_search(user_query, "web_search", final[:1000])
                        
                    except Exception as e:
                        logger.warning(f"Gemini summarization failed, showing raw results: {e}")
                        st.markdown("### 🔍 Search Results (Direct):")
                        st.write(web_results)
                        save_search(user_query, "web_search", web_results[:1000])
                        
                except Exception as e:
                    logger.exception("Web search error: %s", e)
                    st.error(f"❌ Search error: {e}")
    
    elif option == "Weather Check":
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
                        save_search(city, "weather_check", result)
                except Exception as e:
                    logger.exception("Weather fetch failed: %s", e)
                    st.error(f"❌ Error fetching weather: {e}")
    
    elif option == "Smart Assistant":
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

Please provide a well-structured answer:"""
                        
                        response = llm.invoke(prompt)
                        result = response.content if hasattr(response, "content") else str(response)
                        st.markdown("### 🤖 Travel Assistance")
                        st.write(result)
                        save_search(agent_query, "smart_assistant", result[:1000])
                        
                    except Exception as e:
                        st.warning(f"Gemini processing failed: {e}")
                        st.markdown("### 🔍 Raw Results")
                        if weather_info:
                            st.markdown("**🌤 Weather:**")
                            st.write(weather_info)
                        st.markdown("**🔍 Web Search:**")
                        st.write(search_results)
                        save_search(agent_query, "smart_assistant", f"Weather: {weather_info}\nSearch: {search_results}"[:1000])
                        
                except Exception as e:
                    logger.exception("Assistant failed: %s", e)
                    st.error(f"Assistant error: {e}")

# -------------------------
# PAGE: Flight Search
# -------------------------
elif page == "Flight Search":
    st.header("✈️ Flight Search (Amadeus)")
    
    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        st.warning("⚠️ Amadeus API credentials not configured. Please add AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET to your secrets.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        origin = st.text_input("From (Airport Code)", "DEL", max_chars=3).upper()
        destination = st.text_input("To (Airport Code)", "GOI", max_chars=3).upper()
    
    with col2:
        departure_date = st.date_input("Departure Date", datetime.now() + timedelta(days=7))
        return_date = st.date_input("Return Date (Optional)", datetime.now() + timedelta(days=14))
    
    adults = st.number_input("Adults", min_value=1, max_value=9, value=1)
    
    if st.button("Search Flights"):
        if not origin or not destination:
            st.warning("Please enter origin and destination airport codes.")
        else:
            with st.spinner("Searching for flights..."):
                departure_str = departure_date.strftime("%Y-%m-%d")
                return_str = return_date.strftime("%Y-%m-%d") if return_date else None
                
                flights = amadeus_client.search_flights(
                    origin=origin,
                    destination=destination,
                    departure_date=departure_str,
                    return_date=return_str,
                    adults=adults
                )
                
                if "error" in flights:
                    st.error(f"❌ Flight search failed: {flights['error']}")
                elif not flights:
                    st.info("No flights found for your search criteria.")
                else:
                    st.success(f"Found {len(flights)} flights")
                    
                    for i, flight in enumerate(flights):
                        with st.expander(f"Flight {i+1}: {flight['price']} {flight['currency']}"):
                            st.write(f"**Price:** {flight['price']} {flight['currency']}")
                            
                            for j, itinerary in enumerate(flight['itineraries']):
                                st.write(f"**Segment {j+1}** ({itinerary['duration']})")
                                
                                for segment in itinerary['segments']:
                                    dep_time = datetime.fromisoformat(segment['departure']['time'].replace('Z', '+00:00'))
                                    arr_time = datetime.fromisoformat(segment['arrival']['time'].replace('Z', '+00:00'))
                                    
                                    st.write(f"🛫 {segment['departure']['airport']} → 🛬 {segment['arrival']['airport']}")
                                    st.write(f"Time: {dep_time.strftime('%H:%M')} - {arr_time.strftime('%H:%M')}")
                                    st.write(f"Airline: {segment['airline']} Flight {segment['flight_number']}")
                                    st.write("---")

# -------------------------
# PAGE: Itinerary Generator
# -------------------------
elif page == "Itinerary Generator":
    st.header("🗓️ AI Itinerary Generator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        destination = st.text_input("Destination", "Goa, India")
        duration_days = st.number_input("Trip Duration (days)", min_value=1, max_value=30, value=5)
    
    with col2:
        interests = st.multiselect(
            "Your Interests",
            ["Beaches", "History", "Food", "Adventure", "Culture", "Shopping", "Nature", "Nightlife"],
            default=["Beaches", "Food"]
        )
        budget = st.selectbox("Budget", ["low", "medium", "high"])
    
    if st.button("Generate Itinerary"):
        if not destination:
            st.warning("Please enter a destination.")
        else:
            with st.spinner("Creating your personalized itinerary..."):
                itinerary = generate_itinerary(destination, duration_days, interests, budget)
                
                if "error" in itinerary:
                    st.error(f"❌ Itinerary generation failed: {itinerary['error']}")
                else:
                    st.success("✅ Itinerary generated successfully!")
                    st.markdown("### 📅 Your Travel Itinerary")
                    st.write(itinerary['itinerary_text'])
                    
                    # Download option
                    itinerary_text = f"{duration_days}-Day {destination} Itinerary\n\n{itinerary['itinerary_text']}"
                    st.download_button(
                        label="Download Itinerary",
                        data=itinerary_text,
                        file_name=f"{destination.replace(',', '').replace(' ', '_')}_{duration_days}day_itinerary.txt",
                        mime="text/plain"
                    )

# -------------------------
# PAGE: Saved Data
# -------------------------
elif page == "Saved Data":
    st.header("💾 Saved Data")
    
    tab1, tab2, tab3 = st.tabs(["Recent Searches", "Saved Itineraries", "Flight Searches"])
    
    with tab1:
        st.subheader("Recent Searches")
        recent_searches = get_recent_searches(20)
        
        if not recent_searches:
            st.info("No searches saved yet.")
        else:
            for query, search_type, created_at in recent_searches:
                st.write(f"**{search_type.replace('_', ' ').title()}**")
                st.write(f"Query: {query}")
                st.write(f"Time: {created_at}")
                st.write("---")
    
    with tab2:
        st.subheader("Saved Itineraries")
        itineraries = get_saved_itineraries()
        
        if not itineraries:
            st.info("No itineraries saved yet.")
        else:
            for itinerary_id, title, destination, duration_days, created_at in itineraries:
                st.write(f"**{title}**")
                st.write(f"Destination: {destination} | Duration: {duration_days} days")
                st.write(f"Created: {created_at}")
                
                if st.button(f"View Details", key=f"itinerary_{itinerary_id}"):
                    # Here you would fetch and display the full itinerary
                    st.info("Full itinerary details would be displayed here")
                st.write("---")
    
    with tab3:
        st.subheader("Flight Search History")
        # You can implement flight search history display here
        st.info("Flight search history display will be implemented here")

# -------------------------
# PAGE: Document Search (Existing PDF functionality)
# -------------------------
elif page == "Document Search":
    st.header("📄 Document-based Travel Assistant")
    
    # Include the existing PDF upload and search functionality here
    # (Copy the existing document search code from your previous implementation)
    st.info("Document search functionality - to be integrated from existing code")

# -------------------------
# Sidebar with Recent Activity
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("Recent Activity")
    
    recent_searches = get_recent_searches(5)
    if recent_searches:
        for query, search_type, created_at in recent_searches:
            st.caption(f"🔍 {search_type}: {query[:30]}...")
    else:
        st.caption("No recent activity")
    
    st.markdown("---")
    st.info("""
    **Features:**
    - 🔍 Web Search
    - 🌤 Weather Check
    - ✈️ Flight Search
    - 🗓 Itinerary Generator
    - 💾 Search History
    """)