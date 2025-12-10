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

# LangChain + Google Gemini
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import Tool
from PIL import Image
import io

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_ragbot")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Travel Assistant RAGBot", page_icon="🌍", layout="centered")
st.title("🌍 Travel Assistant (RAG + Web Search + Weather + Flights + Hotels + Image Recognition)")
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
        """Load API keys from Streamlit secrets"""
        try:
            if "GOOGLE_API_KEY" not in st.secrets:
                raise ValueError("GOOGLE_API_KEY not found in secrets")
            
            self.keys = {
                'GOOGLE_API_KEY': st.secrets["GOOGLE_API_KEY"],
                'OPENWEATHER_API_KEY': st.secrets.get("OPENWEATHER_API_KEY", ""),
                'AMADEUS_CLIENT_ID': st.secrets.get("AMADEUS_CLIENT_ID", ""),
                'AMADEUS_CLIENT_SECRET': st.secrets.get("AMADEUS_CLIENT_SECRET", ""),
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
            model_names = [model.name for model in models]
            
            validation_results['GOOGLE_API_KEY'] = {
                'valid': True,
                'message': f"✅ Valid ({len(model_names)} models)"
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
# Constants
# -------------------------
PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL_NAME = "models/text-embedding-004"
CHAT_MODEL_CANDIDATES = [
    "gemini-2.0-flash-exp",
    "gemini-2.0-flash",
    "gemini-flash-latest",
    "gemini-pro-latest",
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DB_PATH = "travel_assistant.db"

# -------------------------
# Database Setup
# -------------------------
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            search_type TEXT NOT NULL,
            result_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS flight_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            origin TEXT NOT NULL,
            destination TEXT NOT NULL,
            departure_date TEXT NOT NULL,
            results_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL,
            landmark_name TEXT,
            confidence REAL,
            travel_info TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hotel_searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            destination TEXT NOT NULL,
            check_in TEXT NOT NULL,
            check_out TEXT NOT NULL,
            guests INTEGER DEFAULT 2,
            results_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hotel_favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hotel_name TEXT NOT NULL,
            city TEXT NOT NULL,
            price REAL,
            currency TEXT DEFAULT 'USD',
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_database()

# -------------------------
# Database Helper Functions
# -------------------------
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

def save_flight_search(origin: str, destination: str, departure_date: str, results_count: int = 0):
    """Save flight search to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO flight_searches (origin, destination, departure_date, results_count) VALUES (?, ?, ?, ?)",
        (origin, destination, departure_date, results_count)
    )
    conn.commit()
    conn.close()

def save_image_search(image_name: str, landmark_name: str = None, confidence: float = 0, travel_info: str = ""):
    """Save image search to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO image_searches (image_name, landmark_name, confidence, travel_info) VALUES (?, ?, ?, ?)",
        (image_name, landmark_name, confidence, travel_info[:2000])
    )
    conn.commit()
    conn.close()

def save_hotel_search(destination: str, check_in: str, check_out: str, guests: int, results_count: int = 0):
    """Save hotel search to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO hotel_searches (destination, check_in, check_out, guests, results_count) VALUES (?, ?, ?, ?, ?)",
        (destination, check_in, check_out, guests, results_count)
    )
    conn.commit()
    conn.close()

def save_hotel_favorite(hotel_name: str, city: str, price: float, currency: str = "USD"):
    """Save a hotel to favorites"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO hotel_favorites (hotel_name, city, price, currency, saved_at) 
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (hotel_name, city, price, currency)
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

def get_image_searches(limit: int = 10):
    """Get recent image searches"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT image_name, landmark_name, confidence, created_at FROM image_searches ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return results

def get_hotel_searches(limit: int = 10):
    """Get recent hotel searches"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT destination, check_in, check_out, guests, results_count, created_at 
        FROM hotel_searches 
        ORDER BY created_at DESC LIMIT ?
        """,
        (limit,)
    )
    results = cursor.fetchall()
    conn.close()
    return results

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

# -------------------------
# API Wrappers
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
# Amadeus API Functions
# -------------------------
def get_amadeus_token():
    """Get Amadeus API access token"""
    try:
        amadeus_client_id = api_manager.get_key('AMADEUS_CLIENT_ID')
        amadeus_client_secret = api_manager.get_key('AMADEUS_CLIENT_SECRET')
        
        if not amadeus_client_id or not amadeus_client_secret:
            return None
        
        url = "https://test.api.amadeus.com/v1/security/oauth2/token"
        data = {
            'grant_type': 'client_credentials',
            'client_id': amadeus_client_id,
            'client_secret': amadeus_client_secret
        }
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        
        response = secure_requests_post(url, data=data, headers=headers, api_name="Amadeus", timeout=10)
        
        if response.status_code == 200:
            token_data = response.json()
            return token_data['access_token']
        else:
            logger.error(f"Amadeus token failed: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Failed to get Amadeus token: {e}")
        return None

def get_city_code_amadeus(city_name: str):
    """Get IATA city code from Amadeus"""
    try:
        token = get_amadeus_token()
        if not token:
            return None
        
        headers = {"Authorization": f"Bearer {token}"}
        url = "https://test.api.amadeus.com/v1/reference-data/locations"
        params = {
            "keyword": city_name,
            "subType": "CITY,AIRPORT",
            "page[limit]": 5
        }
        
        response = secure_requests_get(url, headers=headers, params=params, api_name="Amadeus", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("data"):
                for location in data["data"]:
                    if location.get("subType") == "CITY":
                        return location["iataCode"]
                if data["data"]:
                    return data["data"][0].get("iataCode")
        return None
    except Exception as e:
        logger.error(f"City code search error: {e}")
        return None

def search_hotel_offers_amadeus(city_code: str, check_in: str, check_out: str, guests: int = 2):
    """Search for hotels using Amadeus API"""
    try:
        token = get_amadeus_token()
        if not token:
            return {"error": "Failed to authenticate with Amadeus API"}
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Try direct hotel search
        hotel_url = "https://test.api.amadeus.com/v1/reference-data/locations/hotels/by-city"
        hotel_params = {
            "cityCode": city_code,
            "radius": 5,
            "radiusUnit": "KM",
            "hotelSource": "ALL"
        }
        
        logger.info(f"Searching hotels for city code: {city_code}")
        response = requests.get(hotel_url, headers=headers, params=hotel_params, timeout=15)
        
        if response.status_code == 200:
            hotels_data = response.json()
            hotels = hotels_data.get("data", [])
            
            if hotels:
                logger.info(f"✅ Found {len(hotels)} hotels for {city_code}")
                
                # Format the data
                formatted_hotels = []
                for hotel in hotels:
                    formatted_hotel = {
                        "hotel": {
                            "name": hotel.get("name", "Hotel"),
                            "rating": 4.0,
                            "address": hotel.get("address", {}),
                            "description": {
                                "text": f"Hotel located in {city_code}. Book now for your stay from {check_in} to {check_out}."
                            },
                            "amenities": ["Free WiFi", "Restaurant", "Room Service"],
                            "contact": hotel.get("contact", {})
                        },
                        "offers": [{
                            "price": {
                                "total": "150",
                                "currency": "USD"
                            },
                            "room": {
                                "typeEstimated": {
                                    "category": "Standard Room"
                                }
                            },
                            "guests": {
                                "adults": guests
                            }
                        }]
                    }
                    formatted_hotels.append(formatted_hotel)
                
                return {"data": formatted_hotels}
            else:
                logger.warning(f"No hotels found for {city_code}")
                return {"data": []}
        else:
            logger.error(f"Hotel search failed: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Hotel search error: {e}")
        return {"error": f"Hotel search failed: {str(e)}"}

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
        """
        Searches for one-way flights using the Amadeus Flight Offers API.
        """
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

            response = secure_requests_get(url, headers=headers, params=params, api_name="Amadeus", timeout=10)

            if response.status_code == 200:
                data = response.json()
                flights = self._parse_flight_data(data)
                save_flight_search(origin, destination, departure_date, len(flights))
                return flights
            else:
                logger.error(f"Amadeus API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code} - {response.text}"}

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
                'one_way': True,
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

# Initialize Amadeus client if keys are valid
if KEY_VALIDATION['AMADEUS_KEYS']['valid']:
    amadeus_client = AmadeusClient(
        api_manager.get_key('AMADEUS_CLIENT_ID'),
        api_manager.get_key('AMADEUS_CLIENT_SECRET')
    )
else:
    amadeus_client = None

# -------------------------
# LLM and Embedding Functions
# -------------------------
class RateLimiter:
    def __init__(self):
        self.last_call_time = 0
        self.min_interval = 2
    
    def wait_if_needed(self):
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        if time_since_last_call < self.min_interval:
            sleep_time = self.min_interval - time_since_last_call
            time.sleep(sleep_time)
        self.last_call_time = time.time()

rate_limiter = RateLimiter()

def create_embedding_model():
    """Create embedding model"""
    try:
        return GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            google_api_key=api_manager.get_key('GOOGLE_API_KEY')
        )
    except Exception as e:
        logger.error(f"Failed to create embedding model: {e}")
        return None

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
    for attempt in range(max_retries):
        try:
            rate_limiter.wait_if_needed()
            response = llm.invoke(prompt)
            result = response.content if hasattr(response, "content") else str(response)
            api_manager.track_usage('GOOGLE_API_KEY', True)
            return result
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                api_manager.track_usage('GOOGLE_API_KEY', False)
                continue
            else:
                api_manager.track_usage('GOOGLE_API_KEY', False)
                raise e

# -------------------------
# Text Processing Functions
# -------------------------
def split_text_with_meta(text: str):
    """Split text into chunks for processing"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)

def simple_text_search(text: str, query: str, top_k: int = 3) -> List[str]:
    """Simple keyword-based search as fallback"""
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

def build_or_load_vectorstore_from_chunks(chunks: List[str]):
    """Create or load a persisted Chroma vectorstore"""
    try:
        embedding_model = create_embedding_model()
        
        if embedding_model is None:
            return None
            
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            logger.info("Loading existing Chroma vectorstore")
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        else:
            logger.info("Creating new Chroma vectorstore")
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model, persist_directory=PERSIST_DIR)
        return vectorstore
    except Exception as e:
        if "quota" in str(e).lower() or "429" in str(e):
            st.warning("⚠️ Embedding API quota exceeded. Using keyword-based search instead.")
            return None
        else:
            logger.exception("Failed to build/load vectorstore: %s", e)
            raise

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
        
        return itinerary_data
        
    except Exception as e:
        logger.error(f"Itinerary generation error: {e}")
        return generate_basic_itinerary(destination, duration_days, interests, budget)

def generate_basic_itinerary(destination: str, duration_days: int, interests: List[str], budget: str = "medium"):
    """Generate a basic itinerary without AI"""
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
# Tools
# -------------------------
def weather_tool(city: str) -> str:
    """Return weather summary for a city"""
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
    """Run a DuckDuckGoSearchRun search"""
    try:
        ddg = DuckDuckGoSearchRun()
        results = ddg.run(query)
        return results if isinstance(results, str) else str(results)
    except Exception as e:
        logger.exception("Web search failed: %s", e)
        return f"Web search error: {e}"

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
def test_gemini_api_key(api_key: str):
    """Test a Gemini API key to see what models are available"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Get all models
        models = genai.list_models()
        
        # Find vision-capable models
        vision_models = []
        for model in models:
            try:
                if hasattr(model, 'supported_generation_methods'):
                    methods = model.supported_generation_methods
                    if hasattr(methods, '__iter__') and 'generateContent' in methods:
                        vision_models.append({
                            'name': model.name,
                            'description': model.description if hasattr(model, 'description') else '',
                            'methods': list(methods) if hasattr(methods, '__iter__') else str(methods)
                        })
            except:
                continue
        
        return {
            'success': True,
            'total_models': len(models),
            'vision_models': vision_models,
            'all_models': [model.name for model in models[:50]],  # First 50
            'sample_models': [
                {'name': model.name, 'description': getattr(model, 'description', '')}
                for model in models[:10]
            ]
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def check_vision_status():
    """Check and display vision system status"""
    if not vision_client:
        return {"status": "not_configured", "message": "No API key configured"}
    
    status = vision_client.get_status()
    
    if status['vision_available']:
        return {
            "status": "ready",
            "message": f"✅ Vision is ready using {status['model_name']}",
            "details": status
        }
    elif status['model_initialized']:
        return {
            "status": "text_only",
            "message": f"⚠️ Model {status['model_name']} loaded but vision may not be supported",
            "details": status
        }
    elif status['initialization_error']:
        return {
            "status": "error",
            "message": f"❌ Initialization error: {status['initialization_error'][:100]}",
            "details": status
        }
    else:
        return {
            "status": "unavailable",
            "message": "❌ Vision capabilities not available",
            "details": status
        }

# -------------------------
# Image Recognition
# -------------------------
class VisionRecognition:
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.model = None
        self.model_name = None
        self.vision_available = False
        self.initialization_error = None
        
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                
                logger.info("🔧 Initializing Gemini Vision...")
                
                # Get all available models
                try:
                    available_models = genai.list_models()
                    available_model_names = [model.name for model in available_models]
                    logger.info(f"📊 Found {len(available_model_names)} total models")
                    
                    # Log first few models for debugging
                    logger.info("First 5 available models:")
                    for i, model in enumerate(available_models[:5]):
                        logger.info(f"  {i+1}. {model.name}")
                        if hasattr(model, 'supported_generation_methods'):
                            logger.info(f"     Methods: {model.supported_generation_methods}")
                    
                except Exception as list_error:
                    error_msg = f"Failed to list models: {list_error}"
                    logger.error(error_msg)
                    self.initialization_error = error_msg
                    return
                
                # Define models to try in order of preference
                models_to_try = [
                    "gemini-1.5-flash",      # Most reliable and fast
                    "gemini-1.5-pro",        # More capable
                    "gemini-1.0-pro-vision", # Older vision model
                    "gemini-pro-vision",     # Standard vision model
                    "gemini-pro",            # Try non-vision specific
                    "gemini-2.0-flash",      # Experimental/newer
                    "gemini-2.0-flash-exp",  # Experimental
                ]
                
                logger.info(f"🔄 Testing {len(models_to_try)} potential models...")
                
                for model_name in models_to_try:
                    try:
                        # Try both with and without 'models/' prefix
                        variations = [
                            f"models/{model_name}",
                            model_name
                        ]
                        
                        for model_variation in variations:
                            if model_variation in available_model_names:
                                logger.info(f"  Testing: {model_variation}")
                                
                                try:
                                    # Create the model instance
                                    self.model = genai.GenerativeModel(model_variation)
                                    self.model_name = model_variation
                                    
                                    # First test with text-only to verify basic functionality
                                    logger.info(f"    Basic text test...")
                                    text_response = self.model.generate_content("Say 'Hello World'")
                                    
                                    if text_response and hasattr(text_response, 'text'):
                                        logger.info(f"    ✅ Text test passed")
                                        
                                        # Now test vision capability with a simple image
                                        try:
                                            # Create a simple test image in memory
                                            test_img = Image.new('RGB', (100, 100), color=(255, 0, 0))  # Red image
                                            img_bytes = io.BytesIO()
                                            test_img.save(img_bytes, format='PNG')
                                            img_bytes.seek(0)
                                            
                                            # Try to analyze the image
                                            vision_prompt = "What color is this image?"
                                            vision_response = self.model.generate_content(
                                                [vision_prompt, img_bytes.getvalue()]
                                            )
                                            
                                            if vision_response and hasattr(vision_response, 'text'):
                                                self.vision_available = True
                                                logger.info(f"    ✅ Vision test passed!")
                                                logger.info(f"🎉 Successfully loaded vision model: {model_variation}")
                                                
                                                # Test the specific prompt format we'll use
                                                try:
                                                    travel_prompt = "What is this place?"
                                                    test_response = self.model.generate_content(
                                                        [travel_prompt, img_bytes.getvalue()]
                                                    )
                                                    if test_response:
                                                        logger.info(f"    ✅ Travel prompt test passed")
                                                except Exception as prompt_test_error:
                                                    logger.warning(f"    Travel prompt test warning: {prompt_test_error}")
                                                
                                                return  # Success! Exit the loop
                                            
                                        except Exception as vision_test_error:
                                            logger.warning(f"    Vision test failed: {str(vision_test_error)[:100]}")
                                            # This model doesn't support vision, try next one
                                            self.model = None
                                            self.model_name = None
                                            continue
                                    
                                except Exception as model_error:
                                    logger.warning(f"    Model initialization failed: {str(model_error)[:100]}")
                                    self.model = None
                                    self.model_name = None
                                    continue
                                
                    except Exception as e:
                        logger.warning(f"  Error testing {model_name}: {str(e)[:100]}")
                        continue
                
                # If we get here, no vision model worked
                if not self.vision_available:
                    warning_msg = "No vision-capable model found or initialized"
                    logger.warning(warning_msg)
                    
                    # Try a fallback: use any working model (even if not vision)
                    try:
                        logger.info("🔄 Trying fallback to gemini-1.5-flash...")
                        self.model = genai.GenerativeModel('gemini-1.5-flash')
                        self.model_name = 'gemini-1.5-flash'
                        
                        # Quick test
                        test_response = self.model.generate_content("Test")
                        if test_response:
                            logger.info("✅ Fallback model loaded (text-only)")
                            self.initialization_error = "Model loaded but vision may not be supported"
                    except Exception as fallback_error:
                        error_msg = f"Fallback also failed: {fallback_error}"
                        logger.error(error_msg)
                        self.initialization_error = error_msg
                        
            except Exception as e:
                error_msg = f"Failed to initialize Gemini Vision: {e}"
                logger.error(error_msg)
                self.initialization_error = error_msg
    
    def analyze_image(self, image_file):
        """Analyze image using Gemini Vision or fallback to basic analysis"""
        try:
            # Ensure we have a file/bytes to work with
            if image_file is None:
                return {
                    'description': "❌ No image provided",
                    'source': 'error',
                    'vision_available': False
                }
            
            # Read image data
            if hasattr(image_file, 'read'):
                content = image_file.read()
                # Reset file pointer for potential reuse
                image_file.seek(0)
            elif hasattr(image_file, 'getvalue'):
                content = image_file.getvalue()
            else:
                # Assume it's already bytes
                content = image_file
            
            if not content:
                return {
                    'description': "❌ Empty image file",
                    'source': 'error',
                    'vision_available': False
                }
            
            # Open and process image
            try:
                image = Image.open(io.BytesIO(content))
                
                # Convert RGBA to RGB if necessary
                if image.mode == 'RGBA':
                    image = image.convert('RGB')
                
                # Resize if image is too large (to save tokens/bandwidth)
                max_size = (1024, 1024)
                if image.width > max_size[0] or image.height > max_size[1]:
                    image.thumbnail(max_size, Image.Resampling.LANCZOS)
                
            except Exception as img_error:
                logger.error(f"Failed to process image: {img_error}")
                return self._basic_image_analysis_from_file(content)
            
            if self.vision_available and self.model:
                # Use Gemini Vision
                try:
                    # Track API usage
                    api_manager.track_usage('GOOGLE_API_KEY', True)
                    
                    prompt = """
                    Analyze this travel-related image and provide detailed travel information:
                    
                    1. **Identification**: What landmark/place is this? (City, Country)
                    2. **Description**: What makes this place special or famous?
                    3. **Travel Experience**: What can visitors expect to see/do?
                    4. **Best Time to Visit**: Seasonal recommendations
                    5. **Practical Tips**: Entry requirements, costs, travel tips
                    6. **Nearby Attractions**: Other places to visit in the area
                    
                    Format your response with clear headings and bullet points.
                    Be descriptive and helpful for travelers planning a visit.
                    """
                    
                    # Convert image to bytes for Gemini
                    img_bytes = io.BytesIO()
                    image.save(img_bytes, format='PNG')
                    
                    # Generate analysis
                    response = self.model.generate_content([prompt, img_bytes.getvalue()])
                    
                    if response and hasattr(response, 'text'):
                        return {
                            'description': response.text,
                            'source': 'gemini_vision',
                            'model_used': self.model_name,
                            'vision_available': True,
                            'success': True
                        }
                    else:
                        raise Exception("No response text from model")
                        
                except Exception as vision_error:
                    logger.error(f"Gemini Vision analysis failed: {vision_error}")
                    api_manager.track_usage('GOOGLE_API_KEY', False)
                    
                    # Fall back to basic analysis
                    return self._basic_image_analysis(image)
            
            else:
                # Vision not available, use basic analysis
                return self._basic_image_analysis(image)
                
        except Exception as e:
            logger.exception(f"Image analysis failed: {e}")
            api_manager.track_usage('GOOGLE_API_KEY', False)
            
            return {
                'description': f"❌ Error analyzing image: {str(e)[:200]}",
                'source': 'error',
                'vision_available': False,
                'error': str(e)
            }
    
    def _basic_image_analysis(self, image):
        """Basic image analysis when vision API is not available"""
        try:
            image_format = image.format if hasattr(image, 'format') and image.format else "Unknown"
            dimensions = f"{image.width} x {image.height} pixels"
            color_mode = image.mode if hasattr(image, 'mode') else "Unknown"
            
            # Basic image properties analysis
            is_landscape = image.width > image.height * 1.2
            is_square = abs(image.width - image.height) < 10
            aspect = "Landscape" if is_landscape else ("Square" if is_square else "Portrait")
            
            # Estimate if it might be travel-related
            might_be_travel = (
                is_landscape or  # Landscape often means scenery
                image.width > 800 or  # Large images often mean important subjects
                color_mode == 'RGB'  # Color images more likely to be travel photos
            )
            
            analysis = f"""
## 🖼️ Basic Image Analysis

**Image Properties:**
- **Format:** {image_format}
- **Dimensions:** {dimensions}
- **Color Mode:** {color_mode}
- **Aspect Ratio:** {aspect}
- **Size:** Approximately {image.width * image.height / 1000000:.1f} megapixels

**Status:** {'🔴 Gemini Vision API is currently unavailable'}

### 🔍 What can be analyzed:
This appears to be {'a travel-related image' if might_be_travel else 'an image'}. 
Without AI vision capabilities, I can only analyze basic properties.

### 🚀 To enable AI-powered travel analysis:

1. **API Key Configuration:**
   - Ensure your Google API key supports Gemini Vision models
   - Check that "Generative Language API" is enabled in Google Cloud Console

2. **Model Requirements:**
   - You need access to models like `gemini-1.5-flash` or `gemini-pro-vision`
   - Make sure billing is enabled for your Google Cloud project

3. **Quick Fix:**
   - Generate a new API key from [Google AI Studio](https://makersuite.google.com/)
   - Add it to your `.streamlit/secrets.toml` file

### 📋 Manual Identification Tips:
1. **Google Images**: Drag & drop to [images.google.com]
2. **Travel Forums**: Try TripAdvisor or Lonely Planet forums
3. **Reverse Image Search**: Use tools like TinEye
4. **Social Media**: Reddit's r/travel or r/whereisthis

**Note:** To get detailed travel information about this image, Gemini Vision API access is required.
"""
            
            return {
                'description': analysis,
                'source': 'basic_analysis',
                'model_used': 'none',
                'vision_available': False,
                'warning': 'Gemini Vision API not accessible'
            }
            
        except Exception as e:
            logger.error(f"Basic analysis failed: {e}")
            return {
                'description': "❌ Unable to analyze image properties",
                'source': 'error',
                'vision_available': False
            }
    
    def _basic_image_analysis_from_file(self, image_bytes):
        """Basic analysis when image loading fails"""
        return {
            'description': """
## ❌ Image Processing Error

Unable to process the uploaded image. Please check:

### Common Issues:
1. **File Format:** Ensure it's JPG, PNG, or WebP format
2. **File Size:** Image may be too large or corrupted
3. **Permissions:** Check file read permissions
4. **Corruption:** Try a different image file

### ✅ What to try:
- Upload a different image file
- Convert to JPG format if using PNG/WebP
- Reduce image size (under 10MB recommended)
- Check internet connection for API access

**Supported formats:** JPG, PNG, WebP, BMP
**Max size:** 20MB (recommended under 5MB)
""",
            'source': 'error',
            'vision_available': False
        }
    
    def get_status(self):
        """Get the current status of vision capabilities"""
        return {
            'vision_available': self.vision_available,
            'model_name': self.model_name,
            'model_initialized': self.model is not None,
            'api_key_configured': bool(self.gemini_api_key),
            'initialization_error': self.initialization_error
        }
    
    def test_vision_capability(self):
        """Test if vision is working with a simple test image"""
        if not self.vision_available or not self.model:
            return {"success": False, "error": "Vision not available"}
        
        try:
            # Create a simple test image
            test_img = Image.new('RGB', (100, 100), color=(0, 255, 0))  # Green image
            img_bytes = io.BytesIO()
            test_img.save(img_bytes, format='PNG')
            
            prompt = "What color is this square?"
            response = self.model.generate_content([prompt, img_bytes.getvalue()])
            
            if response and hasattr(response, 'text'):
                return {
                    "success": True,
                    "response": response.text[:100],
                    "model": self.model_name
                }
            else:
                return {"success": False, "error": "No response from model"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
# Initialize Vision client
gemini_key = api_manager.get_key('GOOGLE_API_KEY')
vision_client = VisionRecognition(gemini_key) if gemini_key else None

# -------------------------
# Mock Hotel Data for Fallback
# -------------------------
def get_mock_hotel_data(city_name: str = "Sample City"):
    """Provide mock hotel data for testing"""
    return {
        "data": [
            {
                "hotel": {
                    "name": f"Grand {city_name} Hotel",
                    "rating": 4.3,
                    "address": {
                        "cityName": city_name,
                        "lines": ["123 Main Street"],
                        "postalCode": "10001"
                    },
                    "description": {
                        "text": f"A luxurious hotel in the heart of {city_name} with premium amenities."
                    },
                    "amenities": ["Free WiFi", "Swimming Pool", "Fitness Center", "Restaurant", "Spa"]
                },
                "offers": [{
                    "price": {
                        "total": "150",
                        "currency": "USD"
                    },
                    "room": {
                        "typeEstimated": {
                            "category": "Standard Room",
                            "bedType": "King Bed"
                        }
                    },
                    "guests": {
                        "adults": 2
                    }
                }]
            },
            {
                "hotel": {
                    "name": f"{city_name} Central Plaza",
                    "rating": 4.0,
                    "address": {
                        "cityName": city_name,
                        "lines": ["456 Central Avenue"],
                        "postalCode": "10002"
                    },
                    "description": {
                        "text": f"Modern hotel with great city views in downtown {city_name}."
                    },
                    "amenities": ["Free WiFi", "Business Center", "Bar", "Room Service"]
                },
                "offers": [{
                    "price": {
                        "total": "95",
                        "currency": "USD"
                    },
                    "room": {
                        "typeEstimated": {
                            "category": "Deluxe Room",
                            "bedType": "Queen Bed"
                        }
                    },
                    "guests": {
                        "adults": 2
                    }
                }]
            }
        ]
    }

# -------------------------
# Streamlit Navigation
# -------------------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("Go to", [
    "Travel Search", 
    "Flight Search", 
    "Hotel Booking",
    "Itinerary Generator",
    "Document Search",
    "Image Recognition",
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
You are a helpful travel assistant. Please analyze the following search results and provide a comprehensive answer.

SEARCH RESULTS:
{web_results}

USER'S QUESTION: {user_query}

Please provide a detailed answer with these sections:
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
                        st.warning(f"Gemini summarization failed: {e}")
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
                    
                    # Extract city for weather
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

# -------------------------
# PAGE: Flight Search
# -------------------------
elif page == "Flight Search":
    st.header("✈️ Flight Search (One-Way Only)")
    
    if not KEY_VALIDATION['AMADEUS_KEYS']['valid']:
        st.warning("⚠️ Amadeus API credentials not configured or invalid.")
    elif amadeus_client is None:
        st.error("❌ Amadeus client initialization failed.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("From (Airport Code)", "DEL", max_chars=3)
            destination = st.text_input("To (Airport Code)", "GOI", max_chars=3)
        
        with col2:
            departure_date = st.date_input("Departure Date", 
                                          min_value=datetime.now().date(),
                                          value=datetime.now() + timedelta(days=7))
            adults = st.number_input("Number of Passengers", min_value=1, max_value=9, value=1)
        
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
                        
                        flights.sort(key=lambda x: float(x['price']))
                        
                        st.subheader(f"Flights from {origin} to {destination}")
                        
                        for i, flight in enumerate(flights):
                            with st.expander(f"Flight {i+1}: ₹{flight['price']} {flight['currency']}", expanded=(i==0)):
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
    
    use_ai = st.checkbox("Use AI for detailed itinerary", value=True)
    
    if st.button("Generate Itinerary"):
        if not destination:
            st.warning("Please enter a destination.")
        else:
            with st.spinner("Creating your personalized itinerary..."):
                if use_ai:
                    itinerary = generate_itinerary(destination, duration_days, interests, budget)
                else:
                    itinerary = generate_basic_itinerary(destination, duration_days, interests, budget)
                
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
# PAGE: Document Search
# -------------------------
elif page == "Document Search":
    st.header("📄 Document-based Travel Assistant")
    
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
                st.warning("No text extracted from PDF.")
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            st.stop()

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
                            relevant_chunks = simple_text_search(text, user_query, top_k=5)
                            
                            if not relevant_chunks:
                                st.info("📭 No relevant content found in PDF.")
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
                                    st.markdown("### 📖 Answer from Document")
                                    st.write(response)
                                    
                                    with st.expander("View relevant document sections"):
                                        for i, chunk in enumerate(relevant_chunks):
                                            st.markdown(f"**Section {i+1}:**")
                                            st.write(chunk)
                                            
                                except Exception as e:
                                    st.warning(f"AI processing failed: {e}")
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
                                    st.markdown("### 🧭 Answer from PDF")
                                    st.write(response)
                                    
                                    with st.expander("View retrieved document sections"):
                                        for i, doc in enumerate(relevant_docs):
                                            st.markdown(f"**Section {i+1}:**")
                                            st.write(doc.page_content)
                                            
                                except Exception as e:
                                    st.warning(f"LLM failed: {e}")
                                    st.markdown("### 📚 Relevant Document Sections")
                                    for i, doc in enumerate(relevant_docs):
                                        st.markdown(f"**Section {i+1}:**")
                                        st.write(doc.page_content)

                except Exception as e:
                    logger.exception("Document search failed: %s", e)
                    st.error(f"Error: {e}")

# -------------------------
# PAGE: Image Recognition
# -------------------------
elif page == "Image Recognition":
    st.header("🖼️ Image Recognition for Travel")
    
    if not vision_client:
        st.error("❌ Gemini API key not configured. Please add GOOGLE_API_KEY to secrets.")
        st.info("Required: A Gemini API key with vision capabilities")
        st.markdown("""
        ### How to get a Gemini API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Make sure it has access to Gemini Vision models
        4. Add it to your `.streamlit/secrets.toml` file:
        ```
        GOOGLE_API_KEY = "your-api-key-here"
        ```
        """)
    else:
        # Show vision status
        col1, col2 = st.columns([3, 1])
        with col1:
            if vision_client.vision_available:
                st.success(f"✅ Gemini Vision is ready (using {vision_client.model_name})")
            else:
                st.warning("⚠️ Gemini Vision is not available")
        with col2:
            if st.button("🔄 Refresh Status"):
                st.rerun()
        
        # Debug information
        if not vision_client.vision_available:
            with st.expander("🔧 Debug Information"):
                st.write(f"**Gemini Key Configured:** {bool(gemini_key)}")
                st.write(f"**Vision Available:** {vision_client.vision_available}")
                st.write(f"**Model Loaded:** {vision_client.model_name}")
                st.write(f"**Model Initialized:** {vision_client.model is not None}")
                
                # Test API key directly
                if st.button("Test API Key"):
                    try:
                        import google.generativeai as genai
                        genai.configure(api_key=gemini_key)
                        models = genai.list_models()
                        vision_models = [m.name for m in models if "vision" in str(m.supported_generation_methods).lower()]
                        
                        st.write(f"**Available Vision Models:**")
                        if vision_models:
                            for model in vision_models[:5]:  # Show first 5
                                st.write(f"- {model}")
                        else:
                            st.error("No vision models found in your API key")
                    except Exception as e:
                        st.error(f"API Key test failed: {e}")
        
        st.info("📸 Upload an image of a place, landmark, or travel destination to get information about it!")
        
        # Image upload section
        uploaded_image = st.file_uploader(
            "Choose an image file", 
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="Maximum file size: 20MB. Supported formats: JPG, PNG, WebP, BMP"
        )
        
        # Advanced options
        with st.expander("⚙️ Advanced Options"):
            analysis_type = st.selectbox(
                "Analysis Detail Level",
                ["Comprehensive Travel Info", "Landmark Identification", "Quick Analysis", "Custom Prompt"]
            )
            
            custom_prompt = ""
            if analysis_type == "Custom Prompt":
                custom_prompt = st.text_area(
                    "Enter your custom prompt:",
                    value="Analyze this travel image and tell me what you see.",
                    height=100
                )
        
        # Process uploaded image
        if uploaded_image:
            # Display the uploaded image
            col_img1, col_img2 = st.columns([2, 1])
            
            with col_img1:
                st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            with col_img2:
                st.write("**Image Details:**")
                try:
                    image = Image.open(uploaded_image)
                    st.write(f"**Format:** {image.format}")
                    st.write(f"**Dimensions:** {image.width} × {image.height}")
                    st.write(f"**Mode:** {image.mode}")
                    st.write(f"**Size:** {uploaded_image.size / 1024:.1f} KB")
                except:
                    st.write("Unable to read image details")
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    try:
                        # Show progress indicators
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Processing image...")
                        progress_bar.progress(20)
                        
                        # Get the analysis prompt based on selection
                        if analysis_type == "Comprehensive Travel Info":
                            prompt = """
                            Analyze this travel-related image and provide comprehensive travel information:
                            
                            1. **Landmark Identification**: What is this place? Name the landmark, city, and country
                            2. **Historical Significance**: Brief history and cultural importance
                            3. **Travel Experience**: What makes it special for visitors?
                            4. **Best Time to Visit**: Recommended seasons and weather conditions
                            5. **Things to Do**: Activities, tours, and attractions
                            6. **Practical Information**: Entry fees, opening hours, accessibility
                            7. **Nearby Attractions**: Other places to visit in the area
                            8. **Travel Tips**: Local customs, photography rules, safety tips
                            
                            Format with clear headings and use bullet points for lists.
                            """
                        elif analysis_type == "Landmark Identification":
                            prompt = """
                            Identify this landmark/travel destination:
                            
                            1. **Exact Name**: Official name of the place
                            2. **Location**: City, Region, Country
                            3. **Type**: (e.g., Historical site, Natural wonder, Museum, etc.)
                            4. **Brief Description**: What is it known for?
                            5. **UNESCO Status**: If applicable
                            6. **Construction Period**: When was it built/formed?
                            7. **Architectural Style**: If applicable
                            8. **Key Features**: What makes it unique?
                            """
                        elif analysis_type == "Quick Analysis":
                            prompt = """
                            Briefly identify this travel destination:
                            - Name
                            - Location
                            - Main attraction
                            - One interesting fact
                            """
                        else:
                            prompt = custom_prompt
                        
                        status_text.text("Sending to Gemini Vision API...")
                        progress_bar.progress(50)
                        
                        # Analyze image
                        vision_response = vision_client.analyze_image(uploaded_image)
                        
                        status_text.text("Processing results...")
                        progress_bar.progress(80)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Analysis Results")
                        
                        if 'error' in vision_response:
                            st.error(f"❌ Error: {vision_response['error']}")
                        else:
                            # Show source information
                            with st.expander("📋 Analysis Details", expanded=True):
                                if vision_response.get('warning'):
                                    st.warning(vision_response['warning'])
                                
                                if vision_response.get('source') == 'gemini_vision':
                                    st.success("✅ Generated by Gemini Vision AI")
                                    if vision_response.get('model_used'):
                                        st.caption(f"Model: {vision_response.get('model_used')}")
                                elif vision_response.get('source') == 'basic_analysis':
                                    st.info("ℹ️ Basic image analysis (AI Vision not available)")
                                else:
                                    st.info("ℹ️ Manual analysis")
                            
                            # Display the main description
                            if vision_response.get('description'):
                                st.markdown(vision_response['description'])
                            
                            # Save to database
                            try:
                                # Extract potential landmark name from response
                                landmark_name = None
                                description = vision_response.get('description', '')
                                
                                # Simple extraction of potential place names
                                if "taj mahal" in description.lower():
                                    landmark_name = "Taj Mahal"
                                elif "eiffel" in description.lower():
                                    landmark_name = "Eiffel Tower"
                                elif "colosseum" in description.lower():
                                    landmark_name = "Colosseum"
                                elif "great wall" in description.lower():
                                    landmark_name = "Great Wall of China"
                                
                                save_image_search(
                                    image_name=uploaded_image.name,
                                    landmark_name=landmark_name,
                                    confidence=0.8 if vision_response.get('vision_available') else 0.3,
                                    travel_info=vision_response.get('description', '')[:2000]
                                )
                                
                                st.toast("✅ Analysis saved to history", icon="✅")
                            except Exception as save_error:
                                logger.error(f"Failed to save image search: {save_error}")
                        
                        status_text.text("Complete!")
                        progress_bar.progress(100)
                        time.sleep(0.5)
                        progress_bar.empty()
                        status_text.empty()
                        
                    except Exception as e:
                        st.error(f"❌ Image analysis failed: {str(e)}")
                        logger.exception(f"Image analysis error: {e}")
        
        # Image search history
        with st.expander("📚 Recent Image Analyses"):
            recent_searches = get_image_searches(5)
            
            if not recent_searches:
                st.info("No image searches saved yet.")
            else:
                for image_name, landmark_name, confidence, created_at in recent_searches:
                    col_a, col_b = st.columns([3, 1])
                    with col_a:
                        st.write(f"**{image_name}**")
                        if landmark_name:
                            st.write(f"📍 {landmark_name}")
                        if confidence:
                            st.write(f"Confidence: {confidence:.0%}")
                    with col_b:
                        st.write(f"_{created_at}_")
                    st.write("---")
        
        # Tips section
        with st.expander("💡 Tips for Best Results"):
            st.markdown("""
            ### 📸 Image Guidelines:
            1. **Clear & Well-Lit**: Avoid blurry or dark photos
            2. **Landmark Focus**: Center the main subject
            3. **Minimal Crowds**: Fewer people = better recognition
            4. **Daylight Photos**: Natural light works best
            5. **Multiple Angles**: Try different views of the same place
            
            ### 🏆 Best Results With:
            - **Famous Landmarks**: Eiffel Tower, Taj Mahal, Statue of Liberty
            - **Natural Wonders**: Grand Canyon, Niagara Falls
            - **Iconic Buildings**: Sydney Opera House, Burj Khalifa
            - **Historical Sites**: Pyramids, Roman Ruins
            
            ### 🔧 Troubleshooting:
            - **No Results?** Try a clearer image
            - **Wrong Identification?** Upload a different angle
            - **API Errors?** Check your Gemini API key quota
            - **Slow Response?** Large images take longer to process
            """)
        
        # Quick examples
        with st.expander("🎯 Try These Examples"):
            st.markdown("""
            **Test with these famous landmarks (save images from web and upload):**
            
            1. **Taj Mahal, India** - White marble mausoleum
            2. **Eiffel Tower, Paris** - Iron lattice tower
            3. **Great Wall of China** - Stone fortification
            4. **Colosseum, Rome** - Ancient amphitheater
            5. **Statue of Liberty, NYC** - Copper statue
            
            **Natural Wonders:**
            1. **Grand Canyon, USA** - Red rock formations
            2. **Northern Lights** - Aurora in sky
            3. **Mount Everest** - Snow-capped peak
            4. **Great Barrier Reef** - Coral reef underwater
            
            *Note: Upload saved images of these places to test the system.*
            """)
        
        # Footer note
        st.markdown("---")
        st.caption("ℹ️ Powered by Google Gemini Vision AI. Analysis quality depends on image clarity and landmark popularity.")
# -------------------------
# PAGE: API Management
# -------------------------
elif page == "API Management":
    st.header("🔐 API Key Management")
    
    st.subheader("API Key Status")
    
    for key_name, validation in KEY_VALIDATION.items():
        if validation['valid']:
            st.success(f"**{key_name}**: {validation['message']}")
        elif "Not configured" in validation['message']:
            st.warning(f"**{key_name}**: {validation['message']}")
        else:
            st.error(f"**{key_name}**: {validation['message']}")
    
    st.subheader("API Usage Statistics")
    
    usage_stats = api_manager.get_usage_stats()
    if usage_stats:
        for key_name, stats in usage_stats.items():
            if stats['count'] > 0:
                st.write(f"**{key_name}**:")
                st.write(f"  • Total calls: {stats['count']}")
                st.write(f"  • Errors: {stats['errors']}")
                if stats['last_used']:
                    st.write(f"  • Last used: {stats['last_used'].strftime('%Y-%m-%d %H:%M:%S')}")
                st.write("---")
    
    st.subheader("Database API Usage (Last 7 Days)")
    db_stats = get_api_usage_stats(7)
    
    if db_stats:
        for api_name, total_calls, avg_response_time, error_count in db_stats:
            st.write(f"**{api_name}**:")
            st.write(f"  • Total calls: {total_calls}")
            if avg_response_time:
                st.write(f"  • Avg response time: {avg_response_time:.2f}s")
            st.write(f"  • Error rate: {(error_count/total_calls*100):.1f}%" if total_calls > 0 else "  • Error rate: 0%")
            st.write("---")
    else:
        st.info("No API usage data recorded yet.")
    
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
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Recent Searches", 
        "Saved Itineraries", 
        "Flight Searches", 
        "Image Searches",
        "Hotel Searches"
    ])
    
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
                
                if st.button(f"View Details", key=f"view_{itinerary_id}"):
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT itinerary_data FROM itineraries WHERE id = ?", (itinerary_id,))
                    result = cursor.fetchone()
                    conn.close()
                    
                    if result:
                        itinerary_data = json.loads(result[0])
                        st.text_area("Itinerary", itinerary_data.get('itinerary_text', ''), height=300)
                st.write("---")
    
    with tab3:
        st.subheader("Flight Search History")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT origin, destination, departure_date, results_count, created_at FROM flight_searches ORDER BY created_at DESC LIMIT 10")
        flights = cursor.fetchall()
        conn.close()
        
        if not flights:
            st.info("No flight searches saved yet.")
        else:
            for origin, destination, departure_date, results_count, created_at in flights:
                st.write(f"**{origin} → {destination}**")
                st.write(f"Date: {departure_date}")
                st.write(f"Results: {results_count} flights")
                st.write(f"Time: {created_at}")
                st.write("---")
    
    with tab4:
        st.subheader("Image Search History")
        image_searches = get_image_searches(10)
        
        if not image_searches:
            st.info("No image searches saved yet.")
        else:
            for image_name, landmark_name, confidence, created_at in image_searches:
                st.write(f"**Image:** {image_name}")
                if landmark_name:
                    st.write(f"**Landmark:** {landmark_name}")
                    if confidence:
                        st.write(f"**Confidence:** {confidence:.1%}")
                else:
                    st.write("**Status:** No landmark identified")
                st.write(f"**Time:** {created_at}")
                st.write("---")
    
    with tab5:
        st.subheader("Hotel Search History")
        hotel_searches = get_hotel_searches(10)
    
        if not hotel_searches:
            st.info("No hotel searches saved yet.")
        else:
            for destination, check_in, check_out, guests, results_count, created_at in hotel_searches:
                st.write(f"**Destination:** {destination}")
                st.write(f"**Check-in:** {check_in} | **Check-out:** {check_out}")
                st.write(f"**Guests:** {guests} | **Results:** {results_count} hotels")
                st.write(f"**Searched:** {created_at}")
                st.write("---")

# -------------------------
# Sidebar Info
# -------------------------
with st.sidebar:
    st.markdown("---")
    st.subheader("API Status")
    
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
    
    if gemini_key and vision_client and vision_client.vision_available:
        st.success("✅ Gemini Vision")
    elif gemini_key:
        st.info("🤖 Gemini Text")
    else:
        st.info("🤖 No Gemini")
    
    st.markdown("---")
    
    st.info("""
    **Features:**
    - 🔍 Web Search
    - 🌤 Weather
    - ✈️ Flight Search
    - 🏨 Hotel Search
    - 🖼️ Image Recognition
    - 📄 Document RAG
    - 🗓️ Itinerary Generator
    """)
    
    st.markdown("---")
    st.caption(f"v2.0.0 • Last updated: {datetime.now().strftime('%Y-%m-%d')}")