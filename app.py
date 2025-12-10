# app.py
import os
import time
import logging
import sqlite3
import json
import base64
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import requests
import streamlit as st
import pdfplumber

# LangChain + Google Gemini embedding/chat libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import Tool

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("travel_ragbot")

# -------------------------
# Streamlit page config
# -------------------------
st.set_page_config(page_title="Travel Assistant RAGBot", page_icon="🌍", layout="centered")
st.title("🌍 Travel Assistant (RAG + Web Search + Weather + Flights + Image Recognition)")
st.write("Your AI-powered travel companion — Gemini + LangChain + Amadeus + OpenWeather + Vision AI")

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
        
        # Validate Google Gemini API Key (for both chat and vision)
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.keys['GOOGLE_API_KEY'])
            models = genai.list_models()
            model_names = [model.name for model in models]
            
            # Check if Gemini Vision is available
            vision_available = any('gemini' in name.lower() and 'vision' in name.lower() for name in model_names)
            
            validation_results['GOOGLE_API_KEY'] = {
                'valid': True,
                'message': f"✅ Valid ({len(model_names)} models, Vision: {'Yes' if vision_available else 'No'})"
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
EMBEDDING_MODEL_NAME = "models/text-embedding-004"  # ✅ FIXED
# CORRECTED - Each model should be a separate string in the list
CHAT_MODEL_CANDIDATES = [
    "gemini-2.0-flash-exp",        # ✅ Experimental but fast
    "gemini-2.0-flash",            # ✅ Fast and reliable
    "gemini-flash-latest",         # ✅ Latest flash version
    "gemini-pro-latest",           # ✅ Latest pro version
    "gemini-2.5-flash",            # ✅ New 2.5 version
]
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
DB_PATH = "travel_assistant.db"
# Add these to your existing CHAT_MODEL_CANDIDATES or create a separate list
VISION_MODEL_CANDIDATES = [
    "gemini-2.0-flash-exp",      # ✅ Supports vision
    "gemini-2.0-flash",          # ✅ Supports vision
                "gemini-flash-latest",       # ✅ Latest supports vision
                "gemini-pro-latest",         # ✅ Pro supports vision
                "gemini-2.5-flash",          # ✅ 2.5 supports vision
]
# -------------------------
# Rate Limiting and Caching
# -------------------------
class RateLimiter:
    def __init__(self):
        self.last_call_time = 0
        self.min_interval = 2  # seconds between calls
    
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

# Add these functions after your existing database functions (around line 250)

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

# -------------------------
# SQLite Database Setup
# -------------------------
def init_database():
    """Initialize SQLite database for saving searches and itineraries"""
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
            notes TEXT,
            saved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

def save_flight_search(origin: str, destination: str, departure_date: str, results_count: int = 0):
    """Save flight search to database - ONE WAY ONLY"""
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

# Add after your other helper functions (around line 400)

def get_amadeus_token():
    """Get Amadeus API access token (reusable for hotels and flights)"""
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

def search_hotel_offers_amadeus(city_code: str, check_in: str, check_out: str, guests: int = 2):
    """Search for hotel offers using Amadeus API"""
    try:
        token = get_amadeus_token()
        if not token:
            return {"error": "Failed to authenticate with Amadeus API"}
        
        headers = {"Authorization": f"Bearer {token}"}
        
        # Use Hotel Offers API
        url = "https://test.api.amadeus.com/v3/shopping/hotel-offers"
        params = {
            "cityCode": city_code,
            "checkInDate": check_in,
            "checkOutDate": check_out,
            "adults": guests,
            "roomQuantity": 1,
            "bestRateOnly": True,
            "sort": "PRICE",
            "radius": 20,
            "radiusUnit": "KM"
        }
        
        response = secure_requests_get(url, headers=headers, params=params, api_name="Amadeus_Hotels", timeout=15)
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"Hotel search failed: {response.status_code} - {response.text}")
            return {"error": f"API error: {response.status_code}"}
            
    except Exception as e:
        logger.error(f"Hotel search error: {e}")
        return {"error": str(e)}

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
                # Return first city result
                for location in data["data"]:
                    if location.get("subType") == "CITY":
                        return location["iataCode"]
        return None
    except Exception as e:
        logger.error(f"City code search error: {e}")
        return None
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
# Vision Recognition Class - USING GEMINI VISION
# -------------------------
class VisionRecognition:
    def __init__(self, gemini_api_key: str = None):
        self.gemini_api_key = gemini_api_key
        self.model = None
        self.model_name = None
    
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
            
            # Get list of all available models
                available_models = genai.list_models()
                model_names = [model.name for model in available_models]
            
                logger.info(f"Found {len(model_names)} available models")
            
            # Display available models for debugging
                if st.secrets.get("DEBUG", False):
                    st.write("**Available Models:**")
                    for name in model_names[:10]:  # Show first 10
                        st.write(f"- {name}")
            
            # CORRECTED: Prioritize vision-capable Gemini models
            # Using proper model names without "models/" prefix
                vision_model_patterns = [
                 "gemini-2.0-flash-exp",      # ✅ Supports vision
                "gemini-2.0-flash",          # ✅ Supports vision
                "gemini-flash-latest",       # ✅ Latest supports vision
                "gemini-pro-latest",         # ✅ Pro supports vision
                "gemini-2.5-flash",          # ✅ 2.5 supports vision                  # Legacy
                ]
            
                for pattern in vision_model_patterns:
                # Find matching model
                    matching_models = [name for name in model_names if pattern in name]
                
                    if matching_models:
                    # Try each matching model
                        for full_model_name in matching_models:
                            try:
                                logger.info(f"Trying vision model: {full_model_name}")
                            
                            # IMPORTANT: Use just the model name without "models/" prefix
                                model_name_clean = full_model_name.replace("models/", "")
                                self.model = genai.GenerativeModel(model_name_clean)
                                self.model_name = model_name_clean
                            
                            # Quick test with text only to verify it works
                                test_prompt = "Say 'Hello' if you're working"
                                test_response = self.model.generate_content(test_prompt)
                            
                                if test_response and hasattr(test_response, 'text'):
                                    logger.info(f"✅ Vision model {model_name_clean} initialized successfully")
                                    st.sidebar.success(f"Vision: {model_name_clean}")
                                    break
                                else:
                                    self.model = None
                                    self.model_name = None
                            except Exception as e:
                                logger.warning(f"Model {full_model_name} failed: {str(e)[:100]}")
                                self.model = None
                                self.model_name = None
                                continue
                
                    if self.model:
                        break
            
                if not self.model:
                    logger.warning("⚠️ Could not find a working vision-capable Gemini model")
                # Fallback: Try to use any generative model
                    generative_models = [name for name in model_names if 'generateContent' in name.supported_generation_methods]
                
                    if generative_models:
                        try:
                            model_name_clean = generative_models[0].replace("models/", "")
                            self.model = genai.GenerativeModel(model_name_clean)
                            self.model_name = model_name_clean
                            logger.warning(f"⚠️ Using fallback model (may not support vision): {model_name_clean}")
                            st.sidebar.warning(f"Vision: Fallback {model_name_clean}")
                        except:
                            pass
        
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Vision: {e}")
    
    def analyze_image(self, image_file):
        """Analyze image using Gemini Vision API"""
        if not self.model:
            return {"error": "Gemini Vision API not initialized - no working model found"}
        
        try:
            import google.generativeai as genai
            from PIL import Image
            import io
            
            # Read image content
            if hasattr(image_file, 'read'):
                content = image_file.read()
                image_file.seek(0)  # Reset file pointer
            elif isinstance(image_file, str):
                with open(image_file, 'rb') as f:
                    content = f.read()
            else:
                content = image_file.getvalue()
            
            # Create image object
            image = Image.open(io.BytesIO(content))
            
            # Create detailed prompt for travel analysis
            prompt = """
            You are a travel expert analyzing an image. Please provide comprehensive information about this image.
            
            Analyze this travel-related image and provide:
            
            1. **LANDMARK IDENTIFICATION** (if any):
               - Name of landmark/place
               - Type (beach, mountain, temple, building, etc.)
               - Confidence level (High/Medium/Low)
            
            2. **IMAGE DESCRIPTION**:
               - What's visible in the image
               - Key features
               - Setting/location clues
            
            3. **TRAVEL ANALYSIS**:
               - Is this a travel destination?
               - What type of destination?
               - Activities possible here
            
            4. **LOCATION GUESS** (if possible):
               - Country/region
               - City/area
               - Why you think so
            
            5. **TRAVEL RECOMMENDATIONS**:
               - Best time to visit
               - Things to do
               - Travel tips
            
            Format your response with clear sections and bullet points.
            Make it engaging and practical for travelers planning a trip.
            """
            
            # Make API call with rate limiting
            rate_limiter.wait_if_needed()
            
            # Check if model supports images
            try:
                response = self.model.generate_content([prompt, image])
                api_manager.track_usage('GOOGLE_API_KEY', True)
                
                if not response or not hasattr(response, 'text'):
                    return {"error": "No response from Gemini API"}
                
                return self._parse_gemini_response(response.text)
                
            except Exception as e:
                # Model might not support vision, fallback to text-only
                if "image" in str(e).lower() or "vision" in str(e).lower():
                    logger.warning(f"Model {self.model_name} doesn't support images. Using text-only fallback.")
                    return self._text_only_fallback(image, prompt)
                else:
                    raise e
            
        except Exception as e:
            logger.exception(f"Gemini Vision failed: {e}")
            api_manager.track_usage('GOOGLE_API_KEY', False)
            return {"error": str(e)}
    
    def _text_only_fallback(self, image, prompt):
        """Fallback when model doesn't support images"""
        try:
            # Try to extract basic info from image metadata
            from PIL.ExifTags import TAGS
            
            info = {
                'format': image.format,
                'size': image.size,
                'mode': image.mode,
            }
            
            # Try to get EXIF data
            try:
                exif_data = image._getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        info[tag] = str(value)[:100]
            except:
                pass
            
            # Create a basic description based on image properties
            description = f"""
            ## Image Analysis (Basic)
            
            **Image Properties:**
            - Format: {info.get('format', 'Unknown')}
            - Size: {info.get('size', 'Unknown')} pixels
            - Color Mode: {info.get('mode', 'Unknown')}
            
            **Travel Analysis:**
            Since I can't analyze the image content directly, here's general travel advice:
            
            1. **For Clear Outdoor Images:**
               - Likely shows a travel destination
               - Could be a landmark, natural wonder, or cityscape
               - Consider popular destinations that match the image's color palette
            
            2. **Recommendations:**
               - Upload to Google Images reverse search
               - Ask travel communities for identification
               - Check travel photography websites
            
            3. **General Travel Tips:**
               - Research destinations before visiting
               - Check visa requirements
               - Learn basic local phrases
               - Respect local customs and environment
            
            **Note:** For better analysis, ensure:
            - Image is clear and well-lit
            - Landmark is centered and unobstructed
            - File is in JPG or PNG format
            """
            
            return {
                'description': description,
                'landmarks': [],
                'labels': [],
                'travel_info': description,
                'place_name': None,
                'location_guess': None,
                'source': 'text_fallback',
                'warning': 'Vision capabilities not available in current model'
            }
            
        except Exception as e:
            return {"error": f"Text fallback failed: {str(e)}"}
    
    def _parse_gemini_response(self, response_text):
        """Parse Gemini response into structured format"""
        import re
        
        result = {
            'description': response_text,
            'landmarks': [],
            'labels': [],
            'travel_info': response_text,
            'place_name': None,
            'location_guess': None,
            'source': 'gemini_vision',
            'model_used': self.model_name
        }
        
        # Try to extract landmark name using regex patterns
        landmark_patterns = [
            r"(?:This is|I see|That's|It looks like) (?:the )?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:appears to be|seems to be|looks like) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"(?:recognize.*as|identify.*as) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        ]
        
        for pattern in landmark_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                result['place_name'] = match.group(1)
                result['landmarks'].append({
                    'description': match.group(1),
                    'score': 0.7,
                    'source': 'text_analysis'
                })
                break
        
        # Try to extract location mentions
        location_pattern = r"(?:in|from|of|near) ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
        location_matches = re.findall(location_pattern, response_text)
        if location_matches:
            result['location_guess'] = location_matches[0]
        
        # Extract labels/tags from the description
        travel_keywords = ['beach', 'mountain', 'temple', 'church', 'mosque', 'castle', 
                          'palace', 'monument', 'landmark', 'historical', 'tourist',
                          'city', 'town', 'village', 'destination', 'travel', 'vacation']
        
        for keyword in travel_keywords:
            if keyword.lower() in response_text.lower():
                result['labels'].append({
                    'description': keyword,
                    'score': 0.8
                })
        
        return result
    
    def extract_place_info(self, vision_response):
        """Extract travel-relevant information from Vision API response"""
        if 'error' in vision_response:
            return {'error': vision_response['error']}
        
        place_info = {
            'landmarks': vision_response.get('landmarks', []),
            'labels': vision_response.get('labels', []),
            'place_name': vision_response.get('place_name'),
            'location': vision_response.get('location_guess'),
            'description': vision_response.get('description', ''),
            'travel_relevance': self._calculate_travel_relevance(vision_response),
            'source': vision_response.get('source', 'gemini_vision'),
            'model_used': vision_response.get('model_used', 'unknown')
        }
        
        return place_info
    
    def _calculate_travel_relevance(self, vision_response):
        """Calculate how relevant this image is for travel"""
        text = vision_response.get('description', '').lower()
        
        travel_keywords = ['travel', 'tourist', 'destination', 'vacation', 'holiday',
                          'visit', 'sightseeing', 'landmark', 'monument', 'attraction',
                          'beach', 'mountain', 'temple', 'church', 'mosque', 'castle',
                          'historical', 'cultural', 'adventure', 'explore']
        
        score = 0
        for keyword in travel_keywords:
            if keyword in text:
                score += 1
        
        return min(score / len(travel_keywords) * 100, 100)
    
    def generate_travel_info(self, place_info):
        """Generate travel information using the analysis"""
        if 'error' in place_info:
            return f"Error: {place_info['error']}"
        
        # If we already have a good description from Gemini, use it
        if place_info.get('description'):
            return place_info['description']
        
        # Otherwise create basic info
        basic_info = f"""
# Travel Analysis

## What I See:
{place_info.get('description', 'Unable to analyze image in detail.')}

## Travel Relevance: {place_info.get('travel_relevance', 0):.0f}%

## Suggested Actions:
1. **Research online**: Search for similar destinations
2. **Check travel guides**: Look up recommended places
3. **Plan activities**: Based on the type of location
4. **Consult travel forums**: Get real traveler experiences

## Tips:
- Upload clearer images of landmarks for better analysis
- Include famous monuments or natural wonders
- Try different angles and lighting
"""
        
        return basic_info
    
    def get_google_maps_url(self, place_info):
        """Generate Google Maps URL for the place"""
        if place_info.get('place_name'):
            query = place_info['place_name'].replace(' ', '+')
            return f"https://www.google.com/maps/search/{query}"
        
        if place_info.get('location'):
            query = place_info['location'].replace(' ', '+')
            return f"https://www.google.com/maps/search/{query}"
        
        return None

# Initialize Vision client with Gemini Vision
gemini_key = api_manager.get_key('GOOGLE_API_KEY')
vision_client = VisionRecognition(gemini_key) if gemini_key else None

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

def create_chat_llm(temperature: float = 0.3):
    """Create LLM with fallback for different model names"""
    last_error = None
    
    # CORRECTED: Use proper model names that actually exist
    model_candidates  = [
    "gemini-2.0-flash-exp",        # ✅ Experimental but fast
    "gemini-2.0-flash",            # ✅ Fast and reliable
    "gemini-flash-latest",         # ✅ Latest flash version
    "gemini-pro-latest",           # ✅ Latest pro version
    "gemini-2.5-flash",            # ✅ New 2.5 version
]
    
    logger.info(f"Trying models in order: {model_candidates}")
    
    for model_name in model_candidates:
        try:
            logger.info(f"Attempting to initialize model: {model_name}")
            
            # Create the LLM with this specific model name
            llm = ChatGoogleGenerativeAI(
                model=model_name,  # ✅ Pass just the model name string
                temperature=temperature,
                google_api_key=api_manager.get_key('GOOGLE_API_KEY')
            )
            
            # Test if it works
            rate_limiter.wait_if_needed()
            test_response = llm.invoke("Say 'Hello'")
            
            if test_response:
                logger.info(f"✅ Successfully loaded model: {model_name}")
                return llm
                
        except Exception as e:
            last_error = e
            logger.warning(f"❌ Model {model_name} failed: {e}")
            continue
    
    # If all models fail
    error_msg = f"No working model found. Tried: {model_candidates}. Last error: {last_error}"
    logger.error(error_msg)
    st.error(f"❌ {error_msg}")
    
    # Try one more time with a direct call to see available models
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_manager.get_key('GOOGLE_API_KEY'))
        models = genai.list_models()
        available = [m.name.replace("models/", "") for m in models]
        st.info(f"📋 Models available to you: {available}")
    except:
        pass
    
    raise RuntimeError(error_msg)

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
# -------------------------
# Debug Function
# -------------------------
def debug_model_loading():
    """Debug function to see what's happening with model loading"""
    st.write("### 🔍 Debug Model Loading")
    
    # Check what models you actually have
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_manager.get_key('GOOGLE_API_KEY'))
        
        models = genai.list_models()
        st.write("**Available models:**")
        for model in models:
            name = model.name.replace("models/", "")
            methods = model.supported_generation_methods
            if 'generateContent' in methods:
                st.write(f"- {name}")
    
    except Exception as e:
        st.error(f"Error checking models: {e}")

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

# -------------------------
# PAGE: Flight Search (ONE-WAY ONLY)
# -------------------------
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
# PAGE: Image Recognition
# -------------------------
elif page == "Image Recognition":
    st.header("🖼️ Image Recognition for Travel")
    
    st.info("""
    **Upload an image of a place** to get information about it!
    Features:
    • Identify landmarks and places using Gemini Vision
    • Get travel information
    • Discover things to do
    • Find location on map
    """)
    
    # Check if Gemini Vision is configured
    if not vision_client or not gemini_key:
        st.warning("""
        ⚠️ Gemini Vision API not available.
        
        To enable image recognition:
        1. Make sure GOOGLE_API_KEY is configured in Streamlit secrets
        2. The key should have access to Gemini Vision API
        
        Gemini Vision is used for image analysis.
        """)
    else:
        st.success("✅ Gemini Vision API is ready!")
        
        # Image upload section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Choose an image file", 
                type=["jpg", "jpeg", "png", "webp"],
                help="Upload an image of a place, landmark, or travel destination",
                key="vision_upload"
            )
        
        with col2:
            analysis_mode = st.radio(
                "Analysis Mode",
                ["Quick", "Detailed"],
                help="Quick: Basic analysis | Detailed: Full analysis"
            )
        
        if uploaded_image:
            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
            
            # Add analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image with Gemini Vision..."):
                    try:
                        # Analyze image
                        st.info("🔄 Calling Gemini Vision API...")
                        vision_response = vision_client.analyze_image(uploaded_image)
                        
                        if 'error' in vision_response:
                            st.error(f"❌ Gemini Vision Error: {vision_response['error']}")
                        else:
                            # Extract place information
                            place_info = vision_client.extract_place_info(vision_response)
                            
                            # Display results
                            st.markdown("---")
                            st.subheader("📍 Recognition Results")
                            
                            # Show primary landmark
                            if place_info.get('place_name'):
                                st.success(f"**🎯 Identified Place:** {place_info['place_name']}")
                            
                            # Show location guess
                            if place_info.get('location'):
                                st.info(f"**📍 Location Guess:** {place_info['location']}")
                            
                            # Generate travel information
                            st.info("📝 Generating travel information...")
                            travel_info = vision_client.generate_travel_info(place_info)
                            
                            st.markdown("### 🧭 Travel Information")
                            st.markdown(travel_info)
                            
                            # Action buttons
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                maps_url = vision_client.get_google_maps_url(place_info)
                                if maps_url:
                                    st.markdown(f"[🗺️ Open in Google Maps]({maps_url})")
                            
                            with col2:
                                search_query = (place_info.get('place_name') or place_info.get('location') or "travel destination").replace(' ', '+')
                                st.markdown(f"[🔍 Search Online](https://www.google.com/search?q={search_query}+travel)")
                            
                            with col3:
                                if st.button("🗓️ Create Itinerary", key="create_itinerary_btn"):
                                    destination = place_info.get('place_name') or place_info.get('location') or "This Destination"
                                    st.session_state['itinerary_destination'] = destination
                                    st.success(f"Destination '{destination}' saved for itinerary generation!")
                            
                            # Save to database
                            save_image_search(
                                image_name=uploaded_image.name,
                                landmark_name=place_info.get('place_name'),
                                confidence=place_info.get('travel_relevance', 0)/100,
                                travel_info=travel_info[:2000]
                            )
                        
                    except Exception as e:
                        st.error(f"❌ Image analysis failed: {str(e)}")
                        logger.exception("Image recognition error")
        
        # Example images and tips
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            with st.expander("📸 Example Images"):
                st.markdown("""
                **Best Results With:**
                - Famous landmarks (Eiffel Tower, Taj Mahal)
                - Clear, well-lit photos
                - Front-facing views
                - Minimal people/obstructions
                
                **Try Photos Of:**
                - Monuments & historical sites
                - Natural wonders
                - City skylines
                - Cultural landmarks
                """)
        
        with col2:
            with st.expander("💡 Tips for Best Results"):
                st.markdown("""
                1. **Use clear images** - Avoid blurry or dark photos
                2. **Focus on landmarks** - Center the main subject
                3. **Avoid heavy editing** - Filters can confuse the AI
                4. **Include context** - Show surrounding area
                5. **Try multiple angles** - Different views can help
                
                **Powered by:** Gemini 1.5 Flash Vision
                - Can analyze complex images
                - Understands context and relationships
                - Provides detailed descriptions
                """)

# -------------------------
# PAGE: API Management
# -------------------------
elif page == "API Management":
    
    st.header("🔐 API Key Management")
    
    st.info("""
    **Security Note:** API keys are securely stored in Streamlit Secrets.
    Never expose API keys in your code or version control.
    """)
    
    # Display API Key Validation Status
    st.subheader("API Key Status")
    
    for key_name, validation in KEY_VALIDATION.items():
        if validation['valid']:
            st.success(f"**{key_name}**: {validation['message']}")
        elif "Not configured" in validation['message']:
            st.warning(f"**{key_name}**: {validation['message']}")
        else:
            st.error(f"**{key_name}**: {validation['message']}")
    
    # Check Gemini Vision separately
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            models = genai.list_models()
            vision_models = [m for m in models if 'vision' in m.name.lower()]
            if vision_models:
                st.success(f"**GEMINI_VISION**: ✅ Available ({len(vision_models)} vision models)")
            else:
                st.warning("**GEMINI_VISION**: ⚠️ No vision models found in your plan")
        except:
            st.error("**GEMINI_VISION**: ❌ Failed to check vision capability")
    
    # ========== DEBUG BUTTON ADDED HERE ==========
    st.markdown("---")
    st.subheader("🛠️ Debug Tools")
    
    if st.button("Check Available Gemini Models", type="secondary"):
        debug_model_loading()
    # ========== END DEBUG SECTION ==========
    
    # Display API Usage Statistics
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
    
    # Database API Usage Stats
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
    
    # API Key Configuration Guide
    with st.expander("🔧 API Configuration Guide"):
        st.markdown("""
        ### How to Configure API Keys
        
        **1. Google Gemini API (Required for chat & vision):**
        - Visit: https://makersuite.google.com/app/apikey
        - Create new API key
        - Add to Streamlit Secrets as `GOOGLE_API_KEY`
        
        **2. OpenWeather API (Optional):**
        - Visit: https://openweathermap.org/api
        - Sign up for free API key
        - Add to Streamlit Secrets as `OPENWEATHER_API_KEY`
        
        **3. Amadeus API (Optional for flights):**
        - Visit: https://developers.amadeus.com/
        - Create account and new application
        - Add to Streamlit Secrets as:
          - `AMADEUS_CLIENT_ID`
          - `AMADEUS_CLIENT_SECRET`
        
        **Streamlit Secrets Format (.streamlit/secrets.toml):**
        ```toml
        GOOGLE_API_KEY = "your_gemini_key_here"
        OPENWEATHER_API_KEY = "your_weather_key_here"
        AMADEUS_CLIENT_ID = "your_client_id_here"
        AMADEUS_CLIENT_SECRET = "your_client_secret_here"
        ```
        
        **Note:** No Vision API setup needed! Uses Gemini Vision.
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

# ============================
# PAGE: Hotel Booking
# ============================
elif page == "Hotel Booking":
    st.header("🏨 Hotel Booking")
    
    # Check if Amadeus is configured
    if not KEY_VALIDATION['AMADEUS_KEYS']['valid']:
        st.warning("""
        ⚠️ Amadeus API not configured or invalid.
        
        To use hotel booking:
        1. Get Amadeus API keys from https://developers.amadeus.com
        2. Add to Streamlit Secrets:
           - AMADEUS_CLIENT_ID
           - AMADEUS_CLIENT_SECRET
        
        Hotel booking uses the same Amadeus API as flight search.
        """)
        st.stop()
    
    # Create tabs for different hotel features
    tab1, tab2, tab3 = st.tabs(["🔍 Search Hotels", "💾 Saved Hotels", "📊 Hotel Tips"])
    
    with tab1:
        st.subheader("Search Hotels Worldwide")
        
        # Search form
        col1, col2 = st.columns(2)
        
        with col1:
            city = st.text_input(
                "City Name",
                placeholder="e.g., Paris, New York, Tokyo",
                help="Enter the city where you want to stay"
            )
            
            check_in = st.date_input(
                "Check-in Date",
                value=datetime.now() + timedelta(days=7),
                min_value=datetime.now(),
                help="Select your arrival date"
            )
        
        with col2:
            country = st.text_input(
                "Country (Optional)",
                placeholder="e.g., France, USA",
                help="Specify country for more accurate results"
            )
            
            check_out = st.date_input(
                "Check-out Date",
                value=datetime.now() + timedelta(days=14),
                min_value=check_in + timedelta(days=1),
                help="Select your departure date"
            )
        
        # Additional options
        with st.expander("📋 Additional Options"):
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                guests = st.number_input(
                    "Number of Guests",
                    min_value=1,
                    max_value=10,
                    value=2,
                    help="Adults only"
                )
            
            with col_b:
                rooms = st.number_input(
                    "Number of Rooms",
                    min_value=1,
                    max_value=5,
                    value=1
                )
            
            with col_c:
                max_price = st.number_input(
                    "Max Price (USD)",
                    min_value=0,
                    value=500,
                    step=50,
                    help="Maximum price per night"
                )
        
        # Search button
        if st.button("🔍 Search Hotels", type="primary", use_container_width=True):
            if not city.strip():
                st.warning("Please enter a city name.")
            else:
                with st.spinner(f"Searching hotels in {city}..."):
                    try:
                        # Get city code
                        city_code = get_city_code_amadeus(city)
                        
                        if not city_code:
                            st.error(f"❌ Could not find city '{city}' in Amadeus database.")
                            st.info("Try being more specific (e.g., 'New York' instead of 'NYC')")
                        else:
                            # Format dates
                            check_in_str = check_in.strftime("%Y-%m-%d")
                            check_out_str = check_out.strftime("%Y-%m-%d")
                            
                            # Search for hotels
                            st.info(f"📍 Searching hotels in {city} ({city_code})...")
                            
                            hotel_data = search_hotel_offers_amadeus(
                                city_code=city_code,
                                check_in=check_in_str,
                                check_out=check_out_str,
                                guests=guests
                            )
                            
                            if "error" in hotel_data:
                                st.error(f"❌ Hotel search failed: {hotel_data['error']}")
                            elif not hotel_data.get('data'):
                                st.warning(f"No hotels found in {city} for the selected dates.")
                                st.info("Try different dates or a nearby city.")
                            else:
                                hotels = hotel_data['data']
                                st.success(f"✅ Found {len(hotels)} hotels in {city}")
                                
                                # Save search to database
                                save_hotel_search(city, check_in_str, check_out_str, guests, len(hotels))
                                
                                # Display summary
                                st.subheader(f"🏨 Hotels in {city}")
                                
                                # Stats
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                with col_stat1:
                                    st.metric("Hotels Found", len(hotels))
                                with col_stat2:
                                    avg_price = sum([
                                        float(hotel.get('offers', [{}])[0].get('price', {}).get('total', 0))
                                        for hotel in hotels if hotel.get('offers')
                                    ]) / max(len(hotels), 1)
                                    st.metric("Avg. Price", f"${avg_price:.2f}")
                                with col_stat3:
                                    st.metric("Dates", f"{check_in_str} to {check_out_str}")
                                
                                # Display each hotel
                                for i, hotel in enumerate(hotels[:15]):  # Show first 15
                                    hotel_info = hotel.get('hotel', {})
                                    offers = hotel.get('offers', [])
                                    
                                    if offers:
                                        offer = offers[0]
                                        price_info = offer.get('price', {})
                                        price = price_info.get('total', 'N/A')
                                        currency = price_info.get('currency', 'USD')
                                        
                                        # Create expander for each hotel
                                        with st.expander(
                                            f"🏨 {hotel_info.get('name', 'Hotel')} - ${price} {currency}",
                                            expanded=(i < 3)  # First 3 expanded
                                        ):
                                            # Hotel details in columns
                                            col_left, col_right = st.columns([3, 1])
                                            
                                            with col_left:
                                                # Hotel name and rating
                                                st.write(f"**Hotel:** {hotel_info.get('name', 'N/A')}")
                                                
                                                # Rating
                                                if hotel_info.get('rating'):
                                                    rating = hotel_info['rating']
                                                    st.write(f"**Rating:** {rating}/5")
                                                
                                                # Address
                                                if hotel_info.get('address'):
                                                    address = hotel_info['address']
                                                    lines = address.get('lines', [])
                                                    if lines:
                                                        st.write(f"**Address:** {lines[0]}")
                                                    city_name = address.get('cityName', '')
                                                    if city_name:
                                                        st.write(f"**City:** {city_name}")
                                                
                                                # Contact
                                                if hotel_info.get('contact'):
                                                    contact = hotel_info['contact']
                                                    if contact.get('phone'):
                                                        st.write(f"**Phone:** {contact['phone']}")
                                            
                                            with col_right:
                                                # Price and actions
                                                st.write(f"**Price:** ${price} {currency}")
                                                st.write(f"**For:** {guests} guests, {rooms} room(s)")
                                                
                                                # Quick actions
                                                if st.button("💾 Save", key=f"save_{i}", use_container_width=True):
                                                    save_hotel_favorite(
                                                        hotel_info.get('name', 'Hotel'),
                                                        city,
                                                        float(price) if price != 'N/A' else 0,
                                                        currency
                                                    )
                                                    st.success("Hotel saved to favorites!")
                                            
                                            # Hotel description
                                            if hotel_info.get('description'):
                                                st.markdown("**Description:**")
                                                st.write(hotel_info['description']['text'][:300] + "...")
                                            
                                            # Additional details
                                            with st.expander("📋 More Details"):
                                                col_d1, col_d2 = st.columns(2)
                                                
                                                with col_d1:
                                                    # Amenities
                                                    st.write("**Amenities:**")
                                                    amenities = hotel_info.get('amenities', [])
                                                    if amenities:
                                                        for amenity in amenities[:10]:
                                                            st.write(f"• {amenity}")
                                                    else:
                                                        st.write("No amenities listed")
                                                
                                                with col_d2:
                                                    # Booking info
                                                    st.write("**Booking Info:**")
                                                    if offer.get('guests'):
                                                        st.write(f"Guests: {offer['guests'].get('adults', guests)} adults")
                                                    if offer.get('room'):
                                                        st.write(f"Room Type: {offer['room'].get('typeEstimated', {}).get('category', 'Standard')}")
                                                    st.write(f"Check-in: After {hotel_info.get('checkIn', {}).get('time', '14:00')}")
                                                    st.write(f"Check-out: Before {hotel_info.get('checkOut', {}).get('time', '12:00')}")
                                            
                                            # Action buttons
                                            st.markdown("---")
                                            action_cols = st.columns(4)
                                            
                                            with action_cols[0]:
                                                if st.button("📅 Book Now", key=f"book_{i}", use_container_width=True):
                                                    st.info("In a real implementation, this would redirect to booking site")
                                            
                                            with action_cols[1]:
                                                if st.button("📍 View on Map", key=f"map_{i}", use_container_width=True):
                                                    address = hotel_info.get('address', {})
                                                    if address.get('lines'):
                                                        location = f"{city} {address['lines'][0]}".replace(' ', '+')
                                                        st.markdown(f"[Open in Google Maps](https://www.google.com/maps/search/{location})")
                                            
                                            with action_cols[2]:
                                                if st.button("📸 View Photos", key=f"photos_{i}", use_container_width=True):
                                                    st.info("Hotel photos would appear here")
                                            
                                            with action_cols[3]:
                                                if st.button("💰 Price Alert", key=f"alert_{i}", use_container_width=True):
                                                    st.success(f"Price alert set for ${price}!")
                                    
                                    st.markdown("---")  # Separator between hotels
                                
                                # Download results
                                st.download_button(
                                    label="📥 Download Hotel List",
                                    data=json.dumps([h.get('hotel', {}) for h in hotels], indent=2),
                                    file_name=f"hotels_{city}_{check_in_str}.json",
                                    mime="application/json"
                                )
                    
                    except Exception as e:
                        st.error(f"❌ Hotel search error: {str(e)}")
                        logger.exception("Hotel search failed")
    
    with tab2:
        st.subheader("💾 Saved Hotels")
        
        # Get saved hotels from database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT hotel_name, city, price, currency, saved_at FROM hotel_favorites ORDER BY saved_at DESC")
        saved_hotels = cursor.fetchall()
        conn.close()
        
        if not saved_hotels:
            st.info("No hotels saved yet. Search for hotels and click 'Save' to add them here.")
        else:
            st.write(f"You have {len(saved_hotels)} saved hotels:")
            
            for i, (hotel_name, city, price, currency, saved_at) in enumerate(saved_hotels):
                with st.expander(f"🏨 {hotel_name} - {city}"):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Hotel:** {hotel_name}")
                        st.write(f"**Location:** {city}")
                        st.write(f"**Price:** {price} {currency}")
                        st.write(f"**Saved:** {saved_at}")
                    
                    with col2:
                        if st.button("🗑️ Remove", key=f"remove_{i}"):
                            # Remove from database
                            conn = sqlite3.connect(DB_PATH)
                            cursor = conn.cursor()
                            cursor.execute("DELETE FROM hotel_favorites WHERE hotel_name = ? AND city = ?", 
                                         (hotel_name, city))
                            conn.commit()
                            conn.close()
                            st.success("Hotel removed!")
                            st.rerun()
                        
                        if st.button("🔍 Search Again", key=f"search_again_{i}"):
                            st.session_state.hotel_search_city = city
                            st.rerun()
                
                st.write("---")
    
    with tab3:
        st.subheader("📊 Hotel Booking Tips")
        
        tips = [
            "**🎯 Best Booking Times:** Book hotels 1-3 months in advance for best prices",
            "**💰 Price Comparison:** Always check multiple booking sites before confirming",
            "**📅 Flexible Dates:** Mid-week stays are often cheaper than weekends",
            "**🏨 Location Matters:** Hotels near city centers are more expensive but convenient",
            "**⭐ Reviews:** Check recent guest reviews (last 3 months) for accurate info",
            "**🎁 Packages:** Look for hotel+flight packages for better deals",
            "**📱 Mobile Apps:** Booking through hotel apps sometimes offers exclusive discounts",
            "**🔄 Cancellation:** Always check cancellation policies before booking",
            "**🧳 Amenities:** Verify important amenities (WiFi, breakfast, parking) are included",
            "**🗣️ Negotiate:** For longer stays, call the hotel directly for possible discounts"
        ]
        
        for tip in tips:
            st.write(f"• {tip}")
        
        st.markdown("---")
        
        # Quick search suggestions
        st.write("**🔍 Popular Destinations:**")
        popular_cities = ["Paris", "New York", "Tokyo", "London", "Dubai", "Bangkok", "Singapore", "Barcelona"]
        
        cols = st.columns(4)
        for idx, popular_city in enumerate(popular_cities):
            with cols[idx % 4]:
                if st.button(f"🏙️ {popular_city}", key=f"popular_{popular_city}"):
                    st.session_state.hotel_search_city = popular_city
                    st.rerun()

# -------------------------
# PAGE: Saved Data
# -------------------------
elif page == "Saved Data":
    st.header("💾 Saved Data")
    
    tab1, tab2, tab3, tab4 ,tab5 = st.tabs(["Recent Searches", "Saved Itineraries", "Flight Searches", "Image Searches","Hotel Searches"])
    
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
                    # Load full itinerary data
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
    with tab4:
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
            
            # Quick re-search button
                if st.button(f"🔍 Search {destination} again", key=f"hotel_re_{destination}"):
                    st.session_state.hotel_search_city = destination
                    st.rerun()
            
                st.write("---")
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
    
    # Gemini Vision status
    if gemini_key and vision_client and vision_client.model:
        st.success("✅ Gemini Vision")
    elif gemini_key:
        st.info("🖼️ Gemini Vision")
    else:
        st.info("🤖 Gemini Vision")
    
    st.markdown("---")
    
    # Show usage warnings
    usage_stats = api_manager.get_usage_stats()
    if 'GOOGLE_API_KEY' in usage_stats and usage_stats['GOOGLE_API_KEY']['count'] > 0:
        if usage_stats['GOOGLE_API_KEY']['errors'] > 5:
            st.warning(f"⚠️ {usage_stats['GOOGLE_API_KEY']['errors']} API errors")
    
    st.info("""
    **Features:**
    - 🔍 Web Search
    - 🌤 Weather
    - ✈️ Flight Search
    - 🖼️ Image Recognition
    - 📄 Document RAG
    - 🗓️ Itinerary Generator
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
    
    # Show saved itineraries count
    itineraries = get_saved_itineraries()
    if itineraries:
        st.caption(f"📁 {len(itineraries)} saved itineraries")
    
    st.markdown("---")
    st.caption(f"v2.0.0 • Last updated: {datetime.now().strftime('%Y-%m-%d')}")
