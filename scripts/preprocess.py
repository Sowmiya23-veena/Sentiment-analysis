import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load dataset
def load_dataset(file_path):
    # Load the dataset
    data = pd.read_csv(file_path, encoding='latin1', header=None)
    data.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    return data[['sentiment', 'text']]

# Preprocess text
def clean_text(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text.lower()

# Save objects to file
def save_object(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

if __name__ == "__main__":
    # Load the dataset
    file_path = "data/testdata.manual.2009.06.14.csv"
    data = load_dataset(file_path)

    # Clean the text data
    data['text'] = data['text'].apply(clean_text)

    # Map sentiment labels (e.g., 0 = negative, 4 = positive)
    sentiment_map = {0: 0, 4: 1}  # Assuming 0 = negative, 4 = positive
    data['sentiment'] = data['sentiment'].map(sentiment_map)

    # Drop any rows with missing values
    data.dropna(inplace=True)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['text'], data['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Save the vectorizer and datasets
    save_object(vectorizer, 'models/vectorizer.pkl')
    save_object((X_train_vec, X_test_vec, y_train, y_test), 'data/split_data.pkl')

    print("Preprocessing complete. Data saved for training.")
