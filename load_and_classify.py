import re
import string
import pickle

# Data Cleaning and Preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    return text

# Load models and vectorizer
with open('models_and_vectorizer.pkl', 'rb') as f:
    vectorizer, classifiers = pickle.load(f)

# Function to classify a new message
def classify_message(message):
    message = preprocess_text(message)
    message_vectorized = vectorizer.transform([message])
    predictions = {name: clf.predict(message_vectorized)[0] for name, clf in classifiers.items()}
    return predictions

# Example usage
if __name__ == "__main__":
    new_message = input("Enter a message to classify: ")
    predictions = classify_message(new_message)
    for name, prediction in predictions.items():
        print(f"{name}: {prediction}")
