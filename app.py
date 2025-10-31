import os
import requests
import streamlit as st
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.tools import DuckDuckGoSearchRun

# -------------------------
# Load API Keys
# -------------------------
if "GOOGLE_API_KEY" not in st.secrets:
    st.error("❌ GOOGLE_API_KEY not found. Set it in Streamlit Secrets.")
    st.stop()

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

# Directly assign your OpenWeather API key
OPENWEATHER_API_KEY = "db760d35f1b2a58aacc72790fa252bfa"

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Travel Assistant RAGBot", page_icon="🌍", layout="centered")
st.title("🌍 Travel Assistant (RAG + Web Search + Weather)")
st.write("Your AI-powered travel companion — powered by Gemini, LangChain, and OpenWeather!")

# -------------------------
# Ask user for mode
# -------------------------
option = st.radio(
    "How can I assist you today?",
    ("Ask using a travel document", "Search travel info on web", "Check weather in a city")
)

# -------------------------
# CASE 1: RAG Flow (Document Upload)
# -------------------------
if option == "Ask using a travel document":
    uploaded_pdf = st.file_uploader("📄 Upload your travel PDF", type=["pdf"])

    if uploaded_pdf:
        pdf_path = os.path.join(".", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
        st.success(f"✅ Uploaded: {uploaded_pdf.name}")

        # Extract text
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            st.error(f"❌ Failed to read PDF: {e}")
            st.stop()

        # Split text
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )
        chunks = splitter.split_text(text)

        # Embed + create vectorstore
        try:
            embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            vectorstore = Chroma.from_texts(texts=chunks, embedding=embedding_model)
        except Exception as e:
            st.error(f"❌ Embedding error: {e}")
            st.stop()

        # Query section
        st.subheader("💬 Ask something about your document")
        user_query = st.text_input("Enter your question here:")

        if st.button("Get Answer"):
            if user_query.strip():
                try:
                    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                    relevant_docs = retriever.get_relevant_documents(user_query)

                    # -------------------------
                    # Case 1: No relevant chunks → fallback to web search
                    # -------------------------
                    if not relevant_docs:
                        st.info("📭 No relevant information found in your PDF. Searching the web using Gemini...")

                        search_tool = DuckDuckGoSearchRun()
                        search_results = search_tool.run(user_query)

                        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.5)
                        prompt = f"""
You are a travel assistant. Summarize the following search results into a clear, friendly, and useful answer.

Search Results:
{search_results}

Question: {user_query}

Answer:
"""
                        response = llm.invoke(prompt)
                        final_answer = response.content if hasattr(response, "content") else str(response)

                        st.markdown("### 🌐 Web Search Answer:")
                        st.write(final_answer)

                    # -------------------------
                    # Case 2: Found relevant chunks → answer from PDF
                    # -------------------------
                    else:
                        context = "\n\n---\n\n".join(
                            [f"[Chunk {i+1}]: {d.page_content}" for i, d in enumerate(relevant_docs)]
                        )

                        prompt = f"""
You are a friendly travel assistant. Use ONLY the information in the provided context to answer the user's question.
If the answer is not in the context, say: "I cannot answer this based on the provided context."

Context:
{context}

Question: {user_query}

Answer:
"""
                        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.3)
                        response = llm.invoke(prompt)
                        final_answer = response.content if hasattr(response, "content") else str(response)

                        st.markdown("### 🧭 Answer from PDF:")
                        st.write(final_answer)

                        with st.expander("📘 View retrieved document chunks"):
                            for i, doc in enumerate(relevant_docs):
                                st.markdown(f"**Chunk {i+1}:**")
                                st.write(doc.page_content)

                except Exception as e:
                    st.error(f"Error during retrieval/generation: {e}")

# -------------------------
# CASE 2: Web Search using Gemini
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
                search_tool = DuckDuckGoSearchRun()
                search_results = search_tool.run(user_query)

                # Summarize search results using Gemini
                llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-thinking-exp-01-21", temperature=0.5)
                prompt = f"""
You are a travel assistant. Summarize the following search results into a concise, friendly, and informative answer for a traveler.

Search Results:
{search_results}

Question: {user_query}

Answer:
"""
                response = llm.invoke(prompt)
                final_answer = response.content if hasattr(response, "content") else str(response)

                st.markdown("### 🧭 Travel Insights:")
                st.write(final_answer)
            except Exception as e:
                st.error(f"❌ Web search error: {e}")

# -------------------------
# CASE 3: Weather Information (OpenWeather API)
# -------------------------
elif option == "Check weather in a city":
    city = st.text_input("Enter city name:")
    if st.button("Get Weather"):
        if not city.strip():
            st.warning("Please enter a city name.")
        else:
            try:
                url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
                response = requests.get(url)
                data = response.json()

                if data.get("cod") != 200:
                    st.error(f"City not found: {city}")
                else:
                    weather = data["weather"][0]["description"].capitalize()
                    temp = data["main"]["temp"]
                    humidity = data["main"]["humidity"]
                    wind = data["wind"]["speed"]

                    st.markdown(f"### 🌤 Weather in **{city.title()}**")
                    st.write(f"**Condition:** {weather}")
                    st.write(f"**Temperature:** {temp} °C")
                    st.write(f"**Humidity:** {humidity}%")
                    st.write(f"**Wind Speed:** {wind} m/s")
            except Exception as e:
                st.error(f"❌ Error fetching weather: {e}")
