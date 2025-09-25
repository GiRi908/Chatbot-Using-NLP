# NLP Chatbot with Scikit-learn and Streamlit

This project is a machine learning-powered chatbot built using Python. It uses Natural Language Processing (NLP) to understand user intent and provide relevant responses from a predefined knowledge base. The chatbot's core logic is based on a Logistic Regression model for intent classification. The user interface is an interactive web application created with Streamlit.

---

## Features

-   **Intent Recognition**: Understands user queries by classifying them into predefined intents using a TF-IDF Vectorizer and a Logistic Regression model.
-   **Interactive Web UI**: A clean and user-friendly chat interface built with Streamlit, featuring separate pages for the chat, conversation history, and project information.
-   **Conversation Logging**: Automatically saves the conversation history, including user input, the chatbot's response, and a timestamp, to a `chat_log.csv` file.
-   **Conversation History Viewer**: A dedicated tab in the UI allows users to review all past conversations logged in the CSV file.
-   **Modular Knowledge Base**: The chatbot's intelligence, including its intents, user patterns, and potential responses, is stored in an easily editable `intents.json` file.
-   **Model Evaluation**: Includes a standalone script (`chatbot.py`) to train, evaluate the model's performance, and save the model artifacts for future use.

---

## Technologies Used

-   **Python 3.x**
-   **Scikit-learn**: For machine learning (`TfidfVectorizer`, `LogisticRegression`).
-   **Streamlit**: For creating the interactive web application UI.
-   **NLTK (Natural Language Toolkit)**: For NLP tasks, specifically sentence tokenization.
-   **Joblib**: For saving the trained machine learning model to a file.

---

## How It Works

The chatbot operates on a simple yet effective machine learning pipeline:

1.  **Data Loading**: The application starts by loading the predefined intents, patterns, and responses from the `intents.json` file.
2.  **Feature Extraction**: The text `patterns` are converted into a numerical format using the **Term Frequency-Inverse Document Frequency (TF-IDF)** technique. This creates a vector for each pattern that represents the importance of words within it.
3.  **Model Training**: A `Logistic Regression` classifier is trained on the TF-IDF vectors and their corresponding intent `tags`. The model learns to associate specific words and phrases with their correct intent.
4.  **Inference (Chatting)**:
    -   When a user enters a message, it is converted into a TF-IDF vector using the same vectorizer.
    -   The trained model predicts the intent (`tag`) for the user's message.
    -   A random response is selected from the list of responses associated with the predicted tag in `intents.json` and displayed to the user.

---

## Project Structure
```bash
.
├── app.py                  # Main Streamlit application file
├── chatbot.py              # Script for training, evaluating, and saving the model
├── intents.json            # The knowledge base with intents, patterns, and responses
└── README.md               # This file

**Note:** Files like `chat_log.csv`, `vectorizer.pkl`, and `model.pkl` will be generated automatically when you run the scripts.

```

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

-   Ensure you have **Python 3.7** or newer installed on your system.

### 2. Clone the Repository (Optional)

If you have your project in a git repository, clone it. Otherwise, just navigate to your project directory.

```bash
git clone <your-repository-url>
cd Chatbot_Project1
3. Create a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies and avoid conflicts.

Bash

# For Windows
python -m venv venv
.\venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
4. Install Dependencies
Install all the required Python libraries using pip.

Bash

pip install streamlit scikit-learn nltk joblib
5. Download NLTK Data
The scripts require the punkt tokenizer from NLTK. Both app.py and chatbot.py will attempt to download this for you automatically. If you encounter any SSL errors or other issues, you can run the following in a Python interpreter to download it manually:

Python

import nltk
nltk.download('punkt')
How to Run
There are two main scripts you can run.

1. Launch the Chatbot Web Application
This is the primary way to interact with the chatbot. Run the following command in your terminal:

Bash

streamlit run app.py
This will start the web server and open the chatbot interface in a new tab in your default browser. You can start chatting right away!

2. Train and Evaluate the Model Separately (Optional)
If you want to evaluate the model's performance or save the trained model and vectorizer to .pkl files, you can run the chatbot.py script.

Bash

python chatbot.py
This script will:

Split the data into training and testing sets.

Train the Logistic Regression model.

Print a detailed classification report to the console showing the model's accuracy, precision, and recall.

Save vectorizer.pkl and model.pkl to your project directory.

File Descriptions
intents.json: This JSON file is the chatbot's knowledge base. You can easily add new tags (intents), patterns (example user phrases), and responses to expand the chatbot's abilities.

app.py: This is the main application file that runs the Streamlit web interface. It handles the UI, loads data, trains the model in memory, and manages the real-time chat logic and conversation logging.

chatbot.py: A utility script for model development. It is used for training the model on a split dataset, evaluating its performance, and saving the trained model and vectorizer for later use.
