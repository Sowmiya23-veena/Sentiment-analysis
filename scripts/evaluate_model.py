import pickle
from sklearn.metrics import classification_report

# Load datasets, model, and vectorizer
def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Load processed data
    data_path = 'data/split_data.pkl'
    model_path = 'models/sentiment_model.pkl'
    vectorizer_path = 'models/vectorizer.pkl'

    X_train, X_test, y_train, y_test = load_object(data_path)
    model = load_object(model_path)
    vectorizer = load_object(vectorizer_path)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
