import os
import json
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Ensure required libraries are downloaded
nltk.download('punkt')

# Load intents from the JSON file
file_path = os.path.join(os.getcwd(), 'intents.json')  # Update path if needed

try:
    with open(file_path, "r") as file:
        intents = json.load(file)
except FileNotFoundError:
    print(f"Error: The file {file_path} was not found.")
    exit()

# Preprocess the data
tags = []
patterns = []

for intent in intents:
    for pattern in intent.get('patterns', []):
        tags.append(intent['tag'])
        patterns.append(pattern)

# Ensure we have patterns and tags to continue
if not patterns or not tags:
    print("Error: No patterns or tags found in the intents file.")
    exit()

# Create the vectorizer and classifier
vectorizer = TfidfVectorizer()
clf = LogisticRegression(random_state=0, max_iter=10000)

# Split data into training and test sets for evaluation
x_train, x_test, y_train, y_test = train_test_split(patterns, tags, test_size=0.2, random_state=42)

# Convert text data to feature vectors
x_train_vect = vectorizer.fit_transform(x_train)
x_test_vect = vectorizer.transform(x_test)

# Train the model
clf.fit(x_train_vect, y_train)

# Evaluate the model
y_pred = clf.predict(x_test_vect)
print("Model Evaluation:")
print(classification_report(y_test, y_pred))

# Save the trained vectorizer and classifier as pkl files
vectorizer_path = os.path.join(os.getcwd(), 'vectorizer.pkl')
model_path = os.path.join(os.getcwd(), 'model.pkl')

try:
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(clf, model_path)
    print("Model and vectorizer saved successfully!")
except Exception as e:
    print(f"Error saving the model or vectorizer: {e}")
