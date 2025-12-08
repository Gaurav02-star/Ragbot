# app.py
import os
import time
import logging
import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
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
# API Key Manager Class
# -------------------------
class APIKeyManager:
    def __init__(self):
        self.keys = {}
        self.usage_stats = {}
        self.load_keys()
    
    def load_keys(self):
        """Load API keys from Streamlit secrets with validation"""
        try:
            # Required keys
            if "GOOGLE_API_KEY" not in st.secrets:
                raise ValueError("GOOGLE_API_KEY not found in secrets")
            
            self.keys = {
                'GOOGLE_API_KEY': st.secrets["GOOGLE_API_KEY"],
                'OPENWEATHER_API_KEY': st.secrets.get("OPENWEATHER_API_KEY", ""),
                'AMADEUS_CLIENT_ID': st.secrets.get("AMADEUS_CLIENT_ID", ""),
                'AMADEUS_CLIENT_SECRET': st.secrets.get("AMADEUS_CLIENT_SECRET", "")
            }
            
            # Initialize usage stats
            for key in self.keys:
                if self.keys[key]:
                    self.usage_stats[key] = {
                        'count': 0,
                        'last_used': None,
                        'errors': 0
                    }
            
            # Set environment variables
            os.environ["GOOGLE_API_KEY"] = self.keys['GOOGLE_API_KEY']
            
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")
            raise
    
    def validate_keys(self):
        """Validate all API keys and return status"""
        validation_results = {}
        
        # Validate Google Gemini API Key
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.keys['GOOGLE_API_KEY'])
            models = genai.list_models()
            validation_results['GOOGLE_API_KEY'] = {
                'valid': True,
                'message': f"✅ Valid (Available models: {len(list(models))})"
            }
        except Exception as e:
            validation_results['GOOGLE_API_KEY'] = {
                'valid': False,
                'message': f"❌ Invalid: {str(e)[:100]}"
            }
        
        # Validate OpenWeather API Key
        if self.keys['OPENWEATHER_API_KEY']:
            try:
                url = "https://api.openweathermap.org/data/2.5/weather"
                params = {"q": "London", "appid": self.keys['OPENWEATHER_API_KEY']}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    validation_results['OPENWEATHER_API_KEY'] = {
                        'valid': True,
                        'message': "✅ Valid"
                    }
                else:
                    validation_results['OPENWEATHER_API_KEY'] = {
                        'valid': False,
                        'message': f"❌ Invalid: Status {response.status_code}"
                    }
            except Exception as e:
                validation_results['OPENWEATHER_API_KEY'] = {
                    'valid': False,
                    'message': f"❌ Invalid: {str(e)[:100]}"
                }
        else:
            validation_results['OPENWEATHER_API_KEY'] = {
                'valid': False,
                'message': "⚠️ Not configured"
            }
        
        # Validate Amadeus API Keys
        if self.keys['AMADEUS_CLIENT_ID'] and self.keys['AMADEUS_CLIENT_SECRET']:
            try:
                url = "https://test.api.amadeus.com/v1/security/oauth2/token"
                data = {
                    'grant_type': 'client_credentials',
                    'client_id': self.keys['AMADEUS_CLIENT_ID'],
                    'client_secret': self.keys['AMADEUS_CLIENT_SECRET']
                }
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                response = requests.post(url, data=data, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    validation_results['AMADEUS_KEYS'] = {
                        'valid': True,
                        'message': "✅ Valid"
                    }
                else:
                    validation_results['AMADEUS_KEYS'] = {
                        'valid': False,
                        'message': f"❌ Invalid: Status {response.status_code}"
                    }
            except Exception as e:
                validation_results['AMADEUS_KEYS'] = {
                    'valid': False,
                    'message': f"❌ Invalid: {str(e)[:100]}"
                }
        else:
            validation_results['AMADEUS_KEYS'] = {
                'valid': False,
                'message': "⚠️ Not configured"
            }
        
        return validation_results
    
    def track_usage(self, key_name: str, success: bool = True):
        """Track API key usage"""
        if key_name in self.usage_stats:
            self.usage_stats[key_name]['count'] += 1
            self.usage_stats[key_name]['last_used'] = datetime.now()
            if not success:
                self.usage_stats[key_name]['errors'] += 1
    
    def get_usage_stats(self):
        """Get API usage statistics"""
        return self.usage_stats
    
    def get_key(self, key_name: str) -> Optional[str]:
        """Safely get an API key"""
        return self.keys.get(key_name)

# Initialize API Key Manager
try:
    api_manager = APIKeyManager()
    KEY_VALIDATION = api_manager.validate_keys()
except Exception as e:
    st.error(f"❌ Failed to initialize API Key Manager: {e}")
    st.stop()

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
# Rate Limiting and Caching
# -------------------------
class RateLimiter:
    def __init__(self):
        self.last_call_time = 0
        self.min_interval = 7  # seconds between calls (to stay under 10/minute)
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)
        self.last_call_time = time.time()

# Global rate limiter
rate_limiter = RateLimiter()

# Response cache to avoid duplicate API calls
response_cache = {}

def get_cached_response(key: str):
    """Get cached response if available and not expired"""
    if key in response_cache:
        cached_time, response = response_cache[key]
        if time.time() - cached_time < 300:  # 5 minute cache
            return response
    return None

def set_cached_response(key: str, response: Any):
    """Cache a response for 5 minutes"""
    response_cache[key] = (time.time(), response)

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
    
    # Create API usage log table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            api_name TEXT NOT NULL,
            endpoint TEXT,
            status_code INTEGER,
            response_time REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
        (query, search_type, result_text[:1000])
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

def log_api_usage(api_name: str, endpoint: str = "", status_code: int = None, response_time: float = None):
    """Log API usage to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO api_usage (api_name, endpoint, status_code, response_time) VALUES (?, ?, ?, ?)",
        (api_name, endpoint, status_code, response_time)
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

def get_api_usage_stats(days: int = 7):
    """Get API usage statistics for the last N days"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT 
            api_name,
            COUNT(*) as total_calls,
            AVG(response_time) as avg_response_time,
            SUM(CASE WHEN status_code >= 400 THEN 1 ELSE 0 END) as error_count
        FROM api_usage 
        WHERE timestamp >= datetime('now', ?)
        GROUP BY api_name
        """,
        (f'-{days} days',)
    )
    results = cursor.fetchall()
    conn.close()
    return results

# Initialize database
init_database()

# -------------------------
# Secure API Wrappers
# -------------------------
def secure_requests_get(url, params=None, headers=None, api_name="Unknown", timeout=10):
    """Secure wrapper for requests.get with logging"""
    start_time = time.time()
    try:
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response_time = time.time() - start_time
        log_api_usage(api_name, url, response.status_code, response_time)
        api_manager.track_usage(api_name, response.status_code < 400)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        log_api_usage(api_name, url, 0, response_time)
        api_manager.track_usage(api_name, False)
        raise e

def secure_requests_post(url, data=None, json=None, headers=None, api_name="Unknown", timeout=10):
    """Secure wrapper for requests.post with logging"""
    start_time = time.time()
    try:
        response = requests.post(url, data=data, json=json, headers=headers, timeout=timeout)
        response_time = time.time() - start_time
        log_api_usage(api_name, url, response.status_code, response_time)
        api_manager.track_usage(api_name, response.status_code < 400)
        return response
    except Exception as e:
        response_time = time.time() - start_time
        log_api_usage(api_name, url, 0, response_time)
        api_manager.track_usage(api_name, False)
        raise e

# -------------------------
# Amadeus Flight API Integration - ONE WAY ONLY
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
            
            response = secure_requests_post(url, data=data, headers=headers, api_name="Amadeus", timeout=10)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            expires_in = token_data.get('expires_in', 1799) - 300
            self.token_expiry = datetime.now() + timedelta(seconds=expires_in)
            
            return self.access_token
        except Exception as e:
            logger.error(f"Failed to get Amadeus access token: {e}")
            return None
    
    def search_flights(self, origin: str, destination: str, departure_date: str, adults: int = 1):
        """Search for ONE-WAY flights using Amadeus API"""
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
                'max': 10,
                'oneWay': 'true'  # Force one-way flights only
            }
            
            response = secure_requests_get(url, headers=headers, params=params, api_name="Amadeus", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                flights = self._parse_flight_data(data)
                save_flight_search(origin, destination, departure_date, len(flights))
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
                'one_way': True,  # All flights are one-way now
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

# Initialize Amadeus client only if keys are valid
if KEY_VALIDATION['AMADEUS_KEYS']['valid']:
    amadeus_client = AmadeusClient(
        api_manager.get_key('AMADEUS_CLIENT_ID'),
        api_manager.get_key('AMADEUS_CLIENT_SECRET')
    )
else:
    amadeus_client = None
# -------------------------
# Document Search Functions (RAG)
# -------------------------
def simple_text_search(text: str, query: str, top_k: int = 3) -> List[str]:
    """Simple keyword-based search as fallback when embeddings are not available."""
    sentences = []
    for paragraph in text.split('\n'):
        for sentence in paragraph.split('.'):
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:
                sentences.append(sentence)
    
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
    
    scored_sentences.sort(reverse=True, key=lambda x: x[0])
    return [sentence for _, sentence in scored_sentences[:top_k]]

def create_embedding_model():
    """Create embedding model with quota error handling"""
    try:
        embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME)
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
            return None
            
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

def split_text_with_meta(text: str):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# -------------------------
# Enhanced LLM Creation with Better Rate Limiting
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
                google_api_key=api_manager.get_key('GOOGLE_API_KEY')
            )
            rate_limiter.wait_if_needed()
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

def safe_llm_invoke(llm, prompt: str, max_retries: int = 3):
    """Safely invoke LLM with rate limiting and retries"""
    cache_key = f"llm_{hash(prompt)}"
    cached_response = get_cached_response(cache_key)
    if cached_response:
        return cached_response
    
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else str(response)
            set_cached_response(cache_key, result)
            api_manager.track_usage('GOOGLE_API_KEY', True)
            return result
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                st.warning(f"⏳ Rate limit hit. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                api_manager.track_usage('GOOGLE_API_KEY', False)
                continue
            else:
                api_manager.track_usage('GOOGLE_API_KEY', False)
                raise e

# -------------------------
# Itinerary Generation with Fallbacks
# -------------------------
def generate_itinerary(destination: str, duration_days: int, interests: List[str], budget: str = "medium"):
    """Generate a travel itinerary using AI with fallbacks"""
    cache_key = f"itinerary_{destination}_{duration_days}_{'_'.join(interests)}_{budget}"
    cached_response = get_cached_response(cache_key)
    if cached_response:
        return cached_response
    
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
        
        itinerary_text = safe_llm_invoke(llm, prompt)
        
        itinerary_data = {
            'destination': destination,
            'duration_days': duration_days,
            'interests': interests,
            'budget': budget,
            'itinerary_text': itinerary_text,
            'generated_at': datetime.now().isoformat()
        }
        
        title = f"{duration_days}-Day {destination} Trip"
        save_itinerary(title, destination, duration_days, itinerary_data)
        
        set_cached_response(cache_key, itinerary_data)
        return itinerary_data
        
    except Exception as e:
        logger.error(f"Itinerary generation error: {e}")
        return generate_basic_itinerary(destination, duration_days, interests, budget)

def generate_basic_itinerary(destination: str, duration_days: int, interests: List[str], budget: str = "medium"):
    """Generate a basic itinerary without AI when rate limits are hit"""
    st.info("🤖 Using smart itinerary template (AI service limited)")
    
    basic_itinerary = f"""
# {duration_days}-Day {destination} Travel Itinerary

## Traveler Profile
- **Interests**: {', '.join(interests)}
- **Budget**: {budget}
- **Duration**: {duration_days} days

## Quick Travel Tips
1. **Best Time to Visit**: Check local weather patterns
2. **Transport**: Research local transport options
3. **Accommodation**: Book in advance for better rates
4. **Food**: Try local cuisine and street food
5. **Culture**: Respect local customs and traditions

## Suggested Daily Structure
"""
    
    for day in range(1, duration_days + 1):
        basic_itinerary += f"""
### Day {day}
**Morning**: Explore local attractions
**Afternoon**: {', '.join(interests)} activities  
**Evening**: Local dining experience

"""
    
    basic_itinerary += """
## Budget Tips
- Look for free walking tours
- Use public transportation
- Eat at local markets
- Book activities in advance for discounts

## Emergency Contacts
- Local emergency: 112 (most countries)
- Your embassy/consulate
- Travel insurance provider
"""
    
    itinerary_data = {
        'destination': destination,
        'duration_days': duration_days,
        'interests': interests,
        'budget': budget,
        'itinerary_text': basic_itinerary,
        'generated_at': datetime.now().isoformat(),
        'ai_generated': False
    }
    
    title = f"{duration_days}-Day {destination} Trip (Basic)"
    save_itinerary(title, destination, duration_days, itinerary_data)
    
    return itinerary_data

# -------------------------
# Weather Tool with Secure API Calls
# -------------------------
def weather_tool(city: str) -> str:
    """Return a short weather summary for a city using OpenWeatherMap."""
    openweather_key = api_manager.get_key('OPENWEATHER_API_KEY')
    if not openweather_key:
        return "OpenWeather API key not configured."
    
    city = city.strip()
    if not city:
        return "Please provide a city name."

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": openweather_key, "units": "metric"}
        r = secure_requests_get(url, params=params, api_name="OpenWeather", timeout=6)
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
    "Document Search",
    "API Management",
    "Saved Data"
])

# -------------------------
# PAGE: Travel Search
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
                        final = safe_llm_invoke(llm, prompt)
                        st.markdown("### 🧭 Travel Insights")
                        st.write(final)
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
                    
                    city = "Goa"
                    if "weather" in agent_query.lower():
                        for word in agent_query.split():
                            if word.lower() not in ['weather', 'in', 'and', 'the']:
                                city = word
                                break
                    
                    weather_info = ""
                    if "weather" in agent_query.lower():
                        weather_info = weather_tool(city)
                    
                    search_query = agent_query
                    if "weather" in agent_query.lower():
                        search_query = agent_query.replace("weather", "").replace("Weather", "").strip()
                    
                    search_results = web_search_tool(search_query)
                    
                    try:
                        llm = create_chat_llm(temperature=0.3)
                        
                        combined_context = f"WEB SEARCH RESULTS:\n{search_results}"
                        if weather_info:
                            combined_context += f"\n\nWEATHER INFORMATION:\n{weather_info}"
                        
                        prompt = f"""As a knowledgeable travel assistant, provide comprehensive information based on the following data:

{combined_context}

USER'S QUESTION: {agent_query}

Please provide a well-structured answer:"""
                        
                        result = safe_llm_invoke(llm, prompt)
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

elif page == "Flight Search":
    st.header("✈️ Flight Search (One-Way Only)")
    
    if not KEY_VALIDATION['AMADEUS_KEYS']['valid']:
        st.warning("⚠️ Amadeus API credentials not configured or invalid. Please add AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET to your secrets.")
    elif amadeus_client is None:
        st.error("❌ Amadeus client initialization failed. Check your API credentials.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("From (Airport Code)", "DEL", max_chars=3, 
                                  help="Enter 3-letter airport code, e.g., DEL for Delhi, BOM for Mumbai")
            destination = st.text_input("To (Airport Code)", "GOI", max_chars=3,
                                       help="Enter 3-letter airport code, e.g., GOI for Goa, BLR for Bengaluru")
        
        with col2:
            departure_date = st.date_input("Departure Date", 
                                          min_value=datetime.now().date(),
                                          value=datetime.now() + timedelta(days=7))
            adults = st.number_input("Number of Passengers", min_value=1, max_value=9, value=1)
        
        st.info("ℹ️ Searching for one-way flights only")
        
        if st.button("Search Flights"):
            if not origin or not destination:
                st.warning("Please enter origin and destination airport codes.")
            elif origin == destination:
                st.error("Origin and destination cannot be the same.")
            else:
                with st.spinner(f"Searching flights from {origin} to {destination}..."):
                    departure_str = departure_date.strftime("%Y-%m-%d")
                    
                    flights = amadeus_client.search_flights(
                        origin=origin,
                        destination=destination,
                        departure_date=departure_str,
                        adults=adults
                    )
                    
                    if "error" in flights:
                        st.error(f"❌ Flight search failed: {flights['error']}")
                    elif not flights:
                        st.info(f"No one-way flights found from {origin} to {destination} on {departure_date.strftime('%B %d, %Y')}")
                    else:
                        st.success(f"✅ Found {len(flights)} one-way flights")
                        
                        # Sort by price (lowest first)
                        flights.sort(key=lambda x: float(x['price']))
                        
                        # Display summary
                        st.subheader(f"Flights from {origin} to {destination}")
                        st.write(f"**Date:** {departure_date.strftime('%A, %B %d, %Y')}")
                        st.write(f"**Passengers:** {adults} adult(s)")
                        st.write(f"**Total options:** {len(flights)} flights")
                        
                        # Display flights
                        for i, flight in enumerate(flights):
                            with st.expander(f"Flight {i+1}: ₹{flight['price']} {flight['currency']} (One-Way)", expanded=(i==0)):
                                st.write(f"**Price:** ₹{flight['price']} {flight['currency']}")
                                st.write(f"**Type:** One-Way Flight")
                                
                                for j, itinerary in enumerate(flight['itineraries']):
                                    total_duration = itinerary['duration']
                                    st.write(f"**Duration:** {total_duration}")
                                    
                                    for segment in itinerary['segments']:
                                        dep_time = datetime.fromisoformat(segment['departure']['time'].replace('Z', '+00:00'))
                                        arr_time = datetime.fromisoformat(segment['arrival']['time'].replace('Z', '+00:00'))
                                        
                                        col_a, col_b, col_c = st.columns([2, 1, 2])
                                        with col_a:
                                            st.write(f"**Departure:**")
                                            st.write(f"{segment['departure']['airport']}")
                                            st.write(f"{dep_time.strftime('%H:%M')}")
                                        with col_b:
                                            st.write("→")
                                        with col_c:
                                            st.write(f"**Arrival:**")
                                            st.write(f"{segment['arrival']['airport']}")
                                            st.write(f"{arr_time.strftime('%H:%M')}")
                                        
                                        st.write(f"**Airline:** {segment['airline']} Flight {segment['flight_number']}")
                                        st.write("---")
                                
                                # Quick booking info
                                st.caption("ℹ️ Contact airlines directly or visit their website to book this flight")


# -------------------------
# PAGE: Itinerary Generator
# -------------------------
elif page == "Itinerary Generator":
    st.header("🗓️ AI Itinerary Generator")
    
    st.info("💡 **Note**: Free tier has limited AI requests. Basic templates will be used if limits are reached.")
    
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
    
    use_ai = st.checkbox("Use AI for detailed itinerary (may hit rate limits)", value=True)
    
    if st.button("Generate Itinerary"):
        if not destination:
            st.warning("Please enter a destination.")
        else:
            with st.spinner("Creating your personalized itinerary..."):
                if use_ai:
                    itinerary = generate_itinerary(destination, duration_days, interests, budget)
                else:
                    itinerary = generate_basic_itinerary(destination, duration_days, interests, budget)
                
                if "error" in itinerary:
                    st.error(f"❌ Itinerary generation failed: {itinerary['error']}")
                    st.info("🔄 Trying basic itinerary template...")
                    itinerary = generate_basic_itinerary(destination, duration_days, interests, budget)
                else:
                    st.success("✅ Itinerary generated successfully!")
                    
                st.markdown("### 📅 Your Travel Itinerary")
                st.write(itinerary['itinerary_text'])
                
                itinerary_text = f"{duration_days}-Day {destination} Itinerary\n\n{itinerary['itinerary_text']}"
                st.download_button(
                    label="Download Itinerary",
                    data=itinerary_text,
                    file_name=f"{destination.replace(',', '').replace(' ', '_')}_{duration_days}day_itinerary.txt",
                    mime="text/plain"
                )

# -------------------------
# PAGE: Document Search (RAG Functionality)
# -------------------------
elif page == "Document Search":
    st.header("📄 Document-based Travel Assistant")
    st.write("Upload a travel PDF document and ask questions about its content")
    
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
                                    
                                    response = safe_llm_invoke(llm, prompt)
                                    final = response
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
                                    response = safe_llm_invoke(llm, prompt)
                                    final = response
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
# PAGE: API Management (NEW)
# -------------------------
elif page == "API Management":
    st.header("🔐 API Key Management")
    
    st.info("""
    **Security Note:** API keys are securely stored in Streamlit Secrets.
    Never expose API keys in your code or version control.
    """)
    
    # Display API Key Validation Status
    st.subheader("API Key Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for key_name, validation in KEY_VALIDATION.items():
            if validation['valid']:
                st.success(f"**{key_name}**: {validation['message']}")
            elif "Not configured" in validation['message']:
                st.warning(f"**{key_name}**: {validation['message']}")
            else:
                st.error(f"**{key_name}**: {validation['message']}")
    
    # Display API Usage Statistics
    st.subheader("API Usage Statistics")
    
    usage_stats = api_manager.get_usage_stats()
    if usage_stats:
        for key_name, stats in usage_stats.items():
            if stats['count'] > 0:
                st.write(f"**{key_name}**:")
                st.write(f"  • Total calls: {stats['count']}")
                st.write(f"  • Errors: {stats['errors']}")
                st.write(f"  • Last used: {stats['last_used']}")
                st.write("---")
    
    # Database API Usage Stats
    st.subheader("Database API Usage (Last 7 Days)")
    db_stats = get_api_usage_stats(7)
    
    if db_stats:
        for api_name, total_calls, avg_response_time, error_count in db_stats:
            st.write(f"**{api_name}**:")
            st.write(f"  • Total calls: {total_calls}")
            st.write(f"  • Avg response time: {avg_response_time:.2f}s")
            st.write(f"  • Error rate: {(error_count/total_calls*100):.1f}%")
            st.write("---")
    else:
        st.info("No API usage data recorded yet.")
    
    # API Key Configuration Guide
    with st.expander("🔧 API Configuration Guide"):
        st.markdown("""
        ### How to Configure API Keys
        
        **1. Google Gemini API:**
        - Visit: https://makersuite.google.com/app/apikey
        - Create new API key
        - Add to Streamlit Secrets as `GOOGLE_API_KEY`
        
        **2. OpenWeather API:**
        - Visit: https://openweathermap.org/api
        - Sign up for free API key
        - Add to Streamlit Secrets as `OPENWEATHER_API_KEY`
        
        **3. Amadeus API:**
        - Visit: https://developers.amadeus.com/
        - Create account and new application
        - Add to Streamlit Secrets as:
          - `AMADEUS_CLIENT_ID`
          - `AMADEUS_CLIENT_SECRET`
        
        **Streamlit Secrets Format (.streamlit/secrets.toml):**
        ```toml
        GOOGLE_API_KEY = "your_key_here"
        OPENWEATHER_API_KEY = "your_key_here"
        AMADEUS_CLIENT_ID = "your_client_id_here"
        AMADEUS_CLIENT_SECRET = "your_client_secret_here"
        ```
        
        **Security Best Practices:**
        - Never commit secrets.toml to version control
        - Use environment variables in production
        - Rotate API keys regularly
        - Monitor usage to prevent abuse
        """)
    
    # API Health Check
    if st.button("🔄 Run API Health Check"):
        with st.spinner("Checking API health..."):
            new_validation = api_manager.validate_keys()
            
            st.subheader("Health Check Results")
            for key_name, validation in new_validation.items():
                if validation['valid']:
                    st.success(f"✅ **{key_name}**: {validation['message']}")
                elif "Not configured" in validation['message']:
                    st.warning(f"⚠️ **{key_name}**: {validation['message']}")
                else:
                    st.error(f"❌ **{key_name}**: {validation['message']}")

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
                
                # View button for itineraries
                if st.button(f"View Details", key=f"view_{itinerary_id}"):
                    # In a real implementation, you would fetch and display the full itinerary
                    st.info(f"Full itinerary for {title} would be displayed here. Implement full view functionality.")
                st.write("---")
    
    with tab3:
        st.subheader("Flight Search History")
        # Placeholder for flight search history
        st.info("Flight search history display would be implemented here.")

# -------------------------
# Sidebar with Info
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("API Status")
    
    # Show color-coded status
    for key_name, validation in KEY_VALIDATION.items():
        if key_name == 'GOOGLE_API_KEY':
            if validation['valid']:
                st.success("✅ Gemini API")
            else:
                st.error("❌ Gemini API")
        
        elif key_name == 'OPENWEATHER_API_KEY':
            if validation['valid']:
                st.success("✅ Weather API")
            elif "Not configured" in validation['message']:
                st.info("🌤 Weather API")
            else:
                st.error("❌ Weather API")
        
        elif key_name == 'AMADEUS_KEYS':
            if validation['valid']:
                st.success("✅ Amadeus API")
            elif "Not configured" in validation['message']:
                st.info("✈️ Amadeus API")
            else:
                st.error("❌ Amadeus API")
    
    st.markdown("---")
    
    # Show usage warnings
    usage_stats = api_manager.get_usage_stats()
    if 'GOOGLE_API_KEY' in usage_stats and usage_stats['GOOGLE_API_KEY']['count'] > 0:
        if usage_stats['GOOGLE_API_KEY']['errors'] > 5:
            st.warning(f"⚠️ {usage_stats['GOOGLE_API_KEY']['errors']} API errors")
    
    st.info("""
    **Security Features:**
    - 🔐 Encrypted key storage
    - 📊 Usage tracking
    - 🚨 Error monitoring
    - ⚡ Rate limiting
    """)
    
    # Quick stats
    st.markdown("---")
    st.subheader("Quick Stats")
    
    recent_searches = get_recent_searches(5)
    if recent_searches:
        st.write("Recent searches:")
        for query, search_type, created_at in recent_searches[:3]:
            st.caption(f"🔍 {search_type}: {query[:20]}...")
    else:
        st.caption("No recent searches")
    
    st.markdown("---")
    st.caption(f"v1.0.0 • Last updated: {datetime.now().strftime('%Y-%m-%d')}")