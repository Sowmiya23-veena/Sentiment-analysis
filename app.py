import streamlit as st
import pickle
import numpy as np
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from io import BytesIO
from sklearn.decomposition import LatentDirichletAllocation
from gtts import gTTS
import os
import streamlit as st
import requests
import webbrowser

#"6fa00c1a9372251a058e7a56a6ebe7d2"

# Helper function to clean text
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()
def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "audio.mp3"
    tts.save(audio_file)

    # Using Streamlit's built-in audio functionality
    with open(audio_file, "rb") as f:
        st.audio(f.read(), format="audio/mp3")

# Define keywords for each topic
topics_keywords = {
    'Sports': ['football', 'basketball', 'soccer', 'cricket', 'baseball', 'athletics'],
    'Politics': ['election', 'president', 'government', 'policy', 'democracy', 'congress'],
    'Education': ['school', 'university', 'education', 'student', 'teacher', 'exam'],
    'Technology': ['technology', 'ai', 'robotics', 'machine learning', 'coding', 'programming', 'computing'],
    'Health': ['health', 'medicine', 'doctor', 'patient', 'hospital', 'wellness'],
}

# Function to classify trending topics based on keywords
def classify_topic(text):
    text = text.lower()
    topic_count = {topic: 0 for topic in topics_keywords}
    for topic, keywords in topics_keywords.items():
        for keyword in keywords:
            if keyword in text:
                topic_count[topic] += 1
    trending_topic = max(topic_count, key=topic_count.get)
    return trending_topic, topic_count


# Load model and vectorizer
def load_object(filename):
    try:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        print(f"Error: The file {filename} was not found.")
        return None
    except Exception as e:
        print(f"Error loading the file {filename}: {e}")
        return None

# Load the trained model and vectorizer
model = load_object('models/sentiment_model.pkl')
vectorizer = load_object('models/vectorizer.pkl')

def classify_topic(text):
    text = text.lower()
    topic_count = {topic: 0 for topic in topics_keywords}
    for topic, keywords in topics_keywords.items():
        for keyword in keywords:
            if keyword in text:
                topic_count[topic] += 1
    trending_topic = max(topic_count, key=topic_count.get)
    return trending_topic, topic_count


# Streamlit page configuration
st.set_page_config(page_title="Sentiment Analysis", page_icon="💬", layout="wide")
st.markdown("""
    <style>
    .header {color: #3D3D3D; font-size: 50px; font-weight: bold; text-align: center; font-family: Arial, sans-serif;}
    .section-title {color: #0072B2; font-size: 30px; font-family: Arial, sans-serif; margin-top: 20px;}
    .subsection {color: #333333; font-size: 18px; font-family: Arial, sans-serif;}
    .footer {color: #7F8C8D; font-size: 14px; text-align: center; font-family: Arial, sans-serif;}
    .button {background-color: #0072B2; color: white; font-size: 18px; padding: 12px 25px; border-radius: 5px; transition: all 0.3s ease-in-out;}
    .button:hover {background-color: #005a8e; transform: scale(1.05);}
    .stTextArea textarea {font-size: 16px; padding: 12px; border-radius: 5px; transition: all 0.3s ease-in-out;}
    .stTextArea textarea:hover {background-color: #e0e0e0; box-shadow: 0px 0px 5px rgba(0, 112, 178, 0.6);}
    .stTextArea textarea:focus {box-shadow: 0px 0px 5px rgba(0, 112, 178, 0.6);}
    .stButton > button {transition: all 0.3s ease-in-out; border-radius: 10px;}
    .stButton > button:hover {background-color: #005a8e; transform: scale(1.1);}
    .stSlider {border-radius: 12px; transition: all 0.3s ease-in-out;}
    .stSlider:hover {background-color: #f0f0f0;}
    .stSelectbox, .stMultiselect {transition: all 0.3s ease-in-out; border-radius: 10px;}
    .stSelectbox:hover, .stMultiselect:hover {background-color: #f0f0f0;}
    </style>
""", unsafe_allow_html=True)
# Title
st.title("📊 Enhanced Sentiment Analysis & Community Insights")
# Sidebar for instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
    1. Input text into the box below.
    2. Click "Predict" to get the sentiment prediction.
    3. The result will show whether the sentiment is **Positive** or **Negative**.
    4. View trending topics, reports, and actionable insights.
""")
# Add Dark Mode feature in the sidebar
import streamlit as st

# Add Dark Mode feature in the sidebar

# User input section for real-time sentiment analysis
st.header("🌟 Real-Time Sentiment Analysis")
st.subheader("Enter text for real-time sentiment analysis:")

user_input = st.text_area("Text input", height=150, placeholder="Type your text here...")

# Add a spinner to show loading
if st.button("Predict"):
    with st.spinner('Analyzing your sentiment...'):
        if user_input:
            # Clean the text and show the cleaned version
            cleaned_text = clean_text(user_input)
            st.subheader("Preprocessed Text:")
            st.write(cleaned_text)

            # Vectorize the user input
            user_input_vec = vectorizer.transform([cleaned_text])

            # Make prediction and show sentiment confidence
            prediction = model.predict(user_input_vec)
            sentiment = "Positive" if prediction == 1 else "Negative"
            confidence = model.predict_proba(user_input_vec)[0][int(prediction[0])]

            st.subheader(f"Sentiment Prediction: {sentiment}")
            st.write(f"Confidence Score: {confidence * 100:.2f}%")

            # Add Audio Playback for Sentiment Result
            result_text = f"Sentiment: {sentiment}, Confidence: {confidence * 100:.2f}%"
            play_audio(result_text)

            # Visualization (Interactive Word Cloud)
            wordcloud = WordCloud(background_color="white", width=800, height=400).generate(cleaned_text)
            st.image(wordcloud.to_array(), caption=f"{sentiment} Sentiment Word Cloud")

            # Trending Topics
            trending_topic, topic_count = classify_topic(user_input)
            st.subheader(f"Trending Topic: {trending_topic}")
            st.write(f"Topic Breakdown: {topic_count}")
            # Displaying topic count in a bar chart
            topic_df = pd.DataFrame(list(topic_count.items()), columns=["Topic", "Count"])
            st.bar_chart(topic_df.set_index("Topic"))

        else:
            st.error("Please enter some text to analyze.")
st.header("📂 Batch Analysis & Insights")
# Batch prediction and trend analysis
st.subheader("Batch Sentiment Prediction and Trend Analysis")

batch_input = st.text_area("Enter multiple sentences (separated by newlines)", height=200, placeholder="Type each sentence on a new line...")

if st.button("Batch Predict and Analyze"):
    with st.spinner('Processing batch...'):
        if batch_input:
            batch_texts = batch_input.split('\n')
            predictions = []
            sentiments = []
            for text in batch_texts:
                cleaned_text = clean_text(text)
                user_input_vec = vectorizer.transform([cleaned_text])
                prediction = model.predict(user_input_vec)
                sentiment = "Positive" if prediction == 1 else "Negative"
                confidence = model.predict_proba(user_input_vec)[0][int(prediction[0])]
                predictions.append({"text": text, "sentiment": sentiment, "confidence": confidence})
                sentiments.append(sentiment)

            st.subheader("Batch Prediction Results:")
            for result in predictions:
                st.write(f"**Text:** {result['text']}")
                st.write(f"**Sentiment:** {result['sentiment']}")
                st.write(f"**Confidence:** {result['confidence'] * 100:.2f}%\n")

            # Sentiment distribution pie chart
            sentiment_count = Counter(sentiments)
            labels = ['Positive', 'Negative']
            sizes = [sentiment_count['Positive'], sentiment_count['Negative']]

            fig, ax = plt.subplots()
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#4CAF50', '#F44336'])
            ax.axis('equal')
            st.pyplot(fig)

            # Word count visualization
            all_words = ' '.join(batch_texts).split()
            word_counts = Counter(all_words)
            most_common_words = word_counts.most_common(10)

            st.subheader("Word count:")
            for word, count in most_common_words:
                st.write(f"**{word}:** {count} occurrences")

            # Sentiment Report
            positive_percentage = (sentiment_count['Positive'] / len(batch_texts)) * 100
            negative_percentage = (sentiment_count['Negative'] / len(batch_texts)) * 100
            st.subheader("Sentiment Report:")
            st.write(f"Positive Sentiment: {positive_percentage:.2f}%")
            st.write(f"Negative Sentiment: {negative_percentage:.2f}%")

            # Actionable Insights
            st.subheader("Actionable Insights:")
            if positive_percentage > 60:
                st.write("The community is feeling positive. Celebrate the good feedback!")
            elif negative_percentage > 60:
                st.write("The community is facing issues. Immediate action is needed.")
            else:
                st.write("The community sentiment is mixed. Balance improvements and engagement.")

        else:
            st.error("Please enter multiple sentences for batch prediction.")
    
    # Option to download the sentiment report
    def create_sentiment_report(predictions):
        df = pd.DataFrame(predictions)
        buffer = BytesIO()
        df.to_csv(buffer, index=False)
        buffer.seek(0)
        return buffer

    st.download_button(
        label="Download Sentiment Report (CSV)",
        data=create_sentiment_report(predictions),
        file_name="sentiment_report.csv",
        mime="text/csv"
    )
import streamlit as st
import requests

# Function to fetch news from GNews API
def get_gnews_headlines(api_key, query="technology"):
    url = f"https://gnews.io/api/v4/search?q={query}&token={api_key}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Error fetching data: {response.status_code}")
        return None

# Streamlit App Layout and UI Components
st.header("📰 Latest News Updates")
st.title("NEWS FOR CONVENIENCE")

# Ask user for a keyword to search news
query = st.text_input("Enter a keyword to search news", "technology")

# API Key (replace this with your actual API key from GNews)
api_key = '6fa00c1a9372251a058e7a56a6ebe7d2'  # Replace with your GNews API key

# Fetch the news headlines when button is pressed
if st.button("Fetch Latest News"):
    news_data = get_gnews_headlines(api_key, query=query)

    if news_data:
        articles = news_data.get("articles", [])
        if articles:
            st.header(f"Latest News for '{query}'")
            for article in articles:
                st.subheader(article['title'])
                st.write(f"Source: {article['source']['name']}")
                st.write(f"Published on: {article['publishedAt']}")
                st.write(f"[Read more]({article['url']})")
                st.write("------")
        else:
            st.write("No news articles found for your query.")
import streamlit as st
import random
import time

# Initialize session state variables
if "score" not in st.session_state:
    st.session_state["score"] = 0
if "streak" not in st.session_state:
    st.session_state["streak"] = 0
if "random_sentence" not in st.session_state:
    st.session_state["random_sentence"] = ""
if "time_remaining" not in st.session_state:
    st.session_state["time_remaining"] = 30
if "theme" not in st.session_state:
    st.session_state["theme"] = "Light"

# Random sentences
sentences = [
    "I love this app, it's amazing!",
    "This is the worst experience ever.",
    "I feel fantastic today!",
    "I'm so disappointed in this service.",
    "What a great day to be alive!",
    "The food was terrible and cold.",
]

# Sentiment prediction function (mock for demonstration)
def predict_sentiment(text):
    sentiments = ["Positive", "Negative"]
    return random.choice(sentiments)

# Apply custom CSS
def apply_css(theme):
    if theme == "Dark":
        css = """
        <style>
        body {
            background-color: #1e1e1e;
            color: white;
        }
        .stButton > button {
            background-color: #333;
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
        }
        .stRadio > div {
            background-color: #222;
            border-radius: 10px;
            padding: 10px;
        }
        .stProgress > div > div {
            background-color: #66ccff;
        }
        </style>
        """
    else:
        css = """
        <style>
        body {
            background-color: #ffffff;
            color: black;
        }
        .stButton > button {
            background-color: #007BFF;
            color: white;
            border-radius: 8px;
            padding: 10px 15px;
        }
        .stRadio > div {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 10px;
        }
        .stProgress > div > div {
            background-color: #007BFF;
        }
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🎮 Game Settings")
# User Profile
st.sidebar.title("🧑‍💻 Your Profile")
username = st.sidebar.text_input("Enter your name", value="Player")
avatar = st.sidebar.radio("Choose your Avatar", ["👩", "👨", "👾", "🤖", "🧙"])

st.sidebar.write(f"Hello, {avatar} {username}!")


game_mode = st.sidebar.radio("Choose a Game Mode", ["Normal", "Time Challenge", "Streak Challenge"])

# Main Page Title
st.title("🎲 Gamified Sentiment Analysis")
st.write("**Guess the sentiment of sentences and score points!**")

# Random Sentence Generator
if st.button("🔀 Generate Sentence"):
    st.session_state["random_sentence"] = random.choice(sentences)

if st.session_state["random_sentence"]:
    st.subheader("**Sentence:**")
    st.write(st.session_state["random_sentence"])

# User Input
if st.session_state["random_sentence"]:
    user_guess = st.radio("Your Guess:", ["Positive", "Negative"], key="guess")
    if st.button("Submit Guess"):
        actual_sentiment = predict_sentiment(st.session_state["random_sentence"])
        if user_guess == actual_sentiment:
            st.success("🎉 Correct!")
            st.session_state["score"] += 10
            st.session_state["streak"] += 1
        else:
            st.error("❌ Incorrect!")
            st.session_state["streak"] = 0
        st.write(f"**Actual Sentiment:** {actual_sentiment}")
    else:
        st.write("Make a guess and submit!")

# Score and Streak
st.write("### Your Progress")
st.write(f"**Score:** {st.session_state['score']} 🎯")
st.write(f"**Streak:** {st.session_state['streak']} 🔥")
st.progress(st.session_state["score"] % 100)

# Time Challenge Mode
if game_mode == "Time Challenge":
    st.write("⏳ **Time Remaining:**", st.session_state["time_remaining"], "seconds")
    if st.session_state["time_remaining"] > 0:
        st.session_state["time_remaining"] -= 1
    else:
        st.write("⏰ Time's up!")

# Achievements
if st.session_state["streak"] == 5:
    st.balloons()
    st.write("🏆 **Achievement Unlocked: 5 Streak!**")
elif st.session_state["score"] >= 50:
    st.write("🎖️ **Achievement Unlocked: Scored 50 Points!**")

# Leaderboard
st.sidebar.subheader("🏅 Leaderboard (Mock)")
leaderboard = {
    "Player1": 120,
    "Player2": 100,
    "You": st.session_state["score"],
    "Player3": 80,
}
sorted_leaderboard = dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))
for player, score in sorted_leaderboard.items():
    st.sidebar.write(f"{player}: {score}")
