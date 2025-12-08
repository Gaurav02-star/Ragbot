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
# Amadeus Flight API Integration with Secure Wrapper
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
                'max': 10
            }
            
            if return_date:
                params['returnDate'] = return_date
            
            response = secure_requests_get(url, headers=headers, params=params, api_name="Amadeus", timeout=10)
            
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

# Initialize Amadeus client only if keys are valid
if KEY_VALIDATION['AMADEUS_KEYS']['valid']:
    amadeus_client = AmadeusClient(
        api_manager.get_key('AMADEUS_CLIENT_ID'),
        api_manager.get_key('AMADEUS_CLIENT_SECRET')
    )
else:
    amadeus_client = None

# ... [Rest of your code remains the same for Document Search, LLM creation, etc.]
# Just replace the existing API calls with secure_requests_get/post where appropriate

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

# -------------------------
# Streamlit Navigation - Add API Management Page
# -------------------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("Go to", [
    "Travel Search", 
    "Flight Search", 
    "Itinerary Generator",
    "Document Search",
    "API Management",  # NEW PAGE
    "Saved Data"
])

# ... [Rest of your existing pages remain the same until the new API Management page]

# -------------------------
# PAGE: API Management (NEW)
# -------------------------
if page == "API Management":

    st.header("🔐 API Key Management")

    st.info("""
    **Security Note:** API keys are securely stored in Streamlit Secrets.
    Never expose API keys in your code or version control.
    """)

    # --- API Key Status ---
    st.subheader("API Key Status")

    col1, col2 = st.columns(2)

    with col1:
        for key_name, validation in KEY_VALIDATION.items():
            if validation['valid']:
                st.success(f"{key_name}: Valid")
            else:
                st.error(f"{key_name}: Invalid")

    # --- API Usage Statistics ---
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

    # --- Database-level API usage ---
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

# ... [Rest of your existing pages continue]

# -------------------------
# Sidebar with Enhanced API Status
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