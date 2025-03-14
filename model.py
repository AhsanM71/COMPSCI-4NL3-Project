import spacy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset


class TextClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TextClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def lemmatization(texts, allowed_postags=["NOUN", "ADJ", "VERB", "ADV"]):
    """lemmatize text""" 
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    texts_out = []
    
    for text in tqdm(texts):
        doc = nlp(text)
        new_text = []
        for token in doc:
            if token.pos_ in allowed_postags:
                new_text.append(token.lemma_)
        final = " ".join(new_text)
        texts_out.append(final)
    
    return texts_out

def logisticRegressionModel(X_train, y_train, X_test):
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return model

def predict_category_for_logistic_regression(text, model, vectorizer, categories):
    """predict new text category"""
    # first lemmatize
    lemmatized = lemmatization([text])[0]
    # convert to TF-IDF feature
    text_vector = vectorizer.transform([lemmatized])
    
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    prediction_to_index = {-1: 0, 0: 1, 1: 2}
                           
    return {
        'category': categories[prediction_to_index[prediction]],
        'confidence': probability[prediction_to_index[prediction]]
    }

def predict_category_for_neural_network(text, model, vectorizer, categories):
    """Predict new text category using PyTorch model"""
    model.eval()
    lemmatized = lemmatization([text])[0]
    text_vector = vectorizer.transform([lemmatized])
    text_tensor = torch.tensor(text_vector.toarray(), dtype=torch.float32)

    with torch.no_grad():
        output = model(text_tensor)
        probabilities = torch.softmax(output, dim=1)  # Shape: (1, num_classes)
        prediction = torch.argmax(probabilities, dim=1).item()  # Extract single prediction
    
    return {
        'category': categories[prediction],
        'confidence': probabilities[0, prediction].item()  # Correct indexing
    }

def output_results(model, X_train, y_train, X_val, y_val, X_test, y_test, categories):
    train_losses = []
    val_losses = []
    
    for i in range(1, 1001, 100):
        model.set_params(max_iter=i)
        model.fit(X_train, y_train)
        
        y_train_pred_prob = model.predict_proba(X_train)
        y_val_pred_prob = model.predict_proba(X_val)

        train_loss = log_loss(y_train, y_train_pred_prob)
        val_loss = log_loss(y_val, y_val_pred_prob)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 1001, 100), train_losses, label="Training Loss")
    plt.plot(range(1, 1001, 100), val_losses, label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Log Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.show()

    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=categories))
    
def neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories):
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

    y_train_tensor = torch.tensor((y_train + 1).values, dtype=torch.long)
    y_val_tensor = torch.tensor((y_val + 1).values, dtype=torch.long)
    y_test_tensor = torch.tensor((y_test + 1).values, dtype=torch.long)
    
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # The data is loaded in batches of size 32 and shuffled
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # The data is loaded in batches of size 32 and not shuffled (for testing)

    input_dim = X_train.shape[1]  # Number of features (from TF-IDF) -- 2716 in this case
    hidden_dim = 128  # Hidden layer size
    output_dim = len(categories)  # Number of classes

    model = TextClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()   # The most common loss function for classification problems
    optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam is a common optimizer
    
    num_epochs = 10
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation Phase
        model.eval()  # Set the model to evaluation mode
        total_val_loss = 0
        with torch.no_grad():  # Disable gradient computation for validation
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")
    
    return model

def main():
    # test new text
    test_texts = [
        "I'm thrilled with the recent performance of $AAPL! The new product launch exceeded expectations.",
        "The market is unpredictable today. Not sure how $GOOG will move.",
        "Disappointed with $TSLA's quarterly results. The numbers were below projections."
    ]
    df = pd.read_csv("Step3/FinalData/combined.csv")
    # df = pd.read_csv("Step3/FinalData/normalized_data.csv")
    # df = pd.read_csv("Step3/FinalDataGPT/combined.csv")
    texts = df['body']
    labels = df['label']
    categories = ["negative", "neutral", "positive"]
    
    pre_processed_text = lemmatization(texts)
    # print(pre_processed_text)
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        min_df=5,
        preprocessor=lambda x: x,  # keep text unchanged, already preprocessed
        tokenizer=lambda x: x.split()  # simple tokenization, text already standardized
    )
    X_tfidf = tfidf_vectorizer.fit_transform(pre_processed_text)
    
    X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 
    
    # model = logisticRegressionModel(X_train, y_train, X_test)

    # print("\nTesting new texts:")
    # for text in test_texts:
    #     result = predict_category_for_logistic_regression(text, model, tfidf_vectorizer, categories)
    #     print(f"\nText: {text}")
    #     print(f"Category: {result['category']}")
    #     print(f"Confidence: {result['confidence']:.2f}")
    #     print("-" * 50)
        
    # output_results(model, X_train, y_train, X_val,y_val, X_test, y_test, categories)
    
    
    model = neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories)
    
    print("\nTesting new texts:")
    for text in test_texts:
        result = predict_category_for_neural_network(text, model, tfidf_vectorizer, categories)
        print(f"\nText: {text}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}")
    
    
if __name__ == "__main__":
    main()