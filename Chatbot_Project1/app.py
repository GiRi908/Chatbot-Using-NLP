import os
import json
import datetime
import csv
import random
import ssl
import nltk
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Handle SSL issues for NLTK
ssl._create_default_https_context = ssl._create_unverified_context
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Preprocess the data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot logic
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Streamlit interface
def main():
    st.set_page_config(page_title="Chatbot", layout="wide")
    st.title("🤖 Chatbot Application")
    st.sidebar.header("Menu")

    menu = ["Home", "Conversation History", "About"]
    choice = st.sidebar.selectbox("Choose an Option", menu)

    if choice == "Home":
        st.subheader("Welcome to the Chatbot!")
        st.write("Type your message below to start a conversation.")
        
        # Check or create the chat log file
        if not os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['User Input', 'Chatbot Response', 'Timestamp'])

        # Chat interface
        user_input = st.text_input("You:", placeholder="Type your message here...")
        if user_input:
            response = chatbot(user_input)
            st.markdown(f"Chatbot: {response}")

            # Log the conversation
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open('chat_log.csv', 'a', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow([user_input, response, timestamp])

            if response.lower() in ['goodbye', 'bye']:
                st.write("Thank you for chatting with me! Have a great day!")
                st.stop()

    elif choice == "Conversation History":
        st.subheader("📜 Conversation History")
        if os.path.exists('chat_log.csv'):
            with open('chat_log.csv', 'r', encoding='utf-8') as csvfile:
                csv_reader = csv.reader(csvfile)
                next(csv_reader)  # Skip the header
                for row in csv_reader:
                    st.write(f"User: {row[0]}")
                    st.write(f"Chatbot: {row[1]}")
                    st.write(f"Timestamp: {row[2]}")
                    st.markdown("---")
        else:
            st.write("No conversation history found.")

    elif choice == "About":
        st.subheader("About This Project")
        st.write("""
        This chatbot is built using Natural Language Processing (NLP) techniques and Logistic Regression.
        The chatbot identifies user intents and provides responses based on predefined patterns and tags.

        Key Features:
        - Intuitive chatbot interface using Streamlit.
        - Dynamic response generation based on user input.
        - Conversation history saved for review.

        Technologies Used:
        - Python
        - Scikit-learn
        - Streamlit
        - NLTK
        """)

if __name__ == "__main__":
    main()