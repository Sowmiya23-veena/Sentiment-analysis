import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load datasets and vectorizer
def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Load processed data
    data_path = 'data/split_data.pkl'
    vectorizer_path = 'models/vectorizer.pkl'

    X_train, X_test, y_train, y_test = load_object(data_path)
    vectorizer = load_object(vectorizer_path)

    # Train the Logistic Regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Save the trained model
    model_path = 'models/sentiment_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Training complete. Model accuracy: {accuracy:.2f}")
