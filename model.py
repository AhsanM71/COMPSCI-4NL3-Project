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
from nltk.corpus import stopwords
import numpy as np
import re
import subprocess
import nltk
import gensim.downloader as api

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

# def neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories):
#     X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
#     X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

#     y_train_tensor = torch.tensor((y_train + 1).values, dtype=torch.long)
#     y_val_tensor = torch.tensor((y_val + 1).values, dtype=torch.long)
#     y_test_tensor = torch.tensor((y_test + 1).values, dtype=torch.long)
    
#     batch_size = 32
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # The data is loaded in batches of size 32 and shuffled
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # The data is loaded in batches of size 32 and not shuffled (for testing)

#     input_dim = X_train.shape[1]  # Number of features (from TF-IDF) -- 2716 in this case
#     hidden_dim = 128  # Hidden layer size
#     output_dim = len(categories)  # Number of classes

#     model = TextClassifier(input_dim, hidden_dim, output_dim)
#     criterion = nn.CrossEntropyLoss()   # The most common loss function for classification problems
#     optimizer = optim.Adam(model.parameters(), lr=0.001)    # Adam is a common optimizer
    
#     num_epochs = 10
#     train_losses = []
#     val_losses = []

#     for epoch in range(num_epochs):
#         # Training Phase
#         model.train()
#         total_train_loss = 0
#         for batch_X, batch_y in train_loader:
#             optimizer.zero_grad()
#             outputs = model(batch_X)
#             loss = criterion(outputs, batch_y)
#             loss.backward()
#             optimizer.step()
#             total_train_loss += loss.item()
        
#         avg_train_loss = total_train_loss / len(train_loader)
#         train_losses.append(avg_train_loss)
        
#         # Validation Phase
#         model.eval()  # Set the model to evaluation mode
#         total_val_loss = 0
#         with torch.no_grad():  # Disable gradient computation for validation
#             for batch_X, batch_y in test_loader:
#                 outputs = model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 total_val_loss += loss.item()
        
#         avg_val_loss = total_val_loss / len(test_loader)
#         val_losses.append(avg_val_loss)
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

#     plt.figure(figsize=(8, 5))
#     plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
#     plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linestyle='--')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training vs Validation Loss')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
    
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for batch_X, batch_y in test_loader:
#             outputs = model(batch_X)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == batch_y).sum().item()
#             total += batch_y.size(0)

#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy:.2f}")
    
#     return model, test_loader

def neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories):
    # Convert input data to tensors
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

    # Convert labels to tensors (Shifted back to -1, 0, 1)
    y_train_tensor = torch.tensor((y_train+1).values, dtype=torch.long)  
    y_val_tensor = torch.tensor((y_val+1).values, dtype=torch.long)
    y_test_tensor = torch.tensor((y_test+1).values, dtype=torch.long)

    # Create datasets and data loaders
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model parameters
    input_dim = X_train.shape[1]  # Number of features (e.g., 2716)
    hidden_dim = 128  # Can be tuned
    output_dim = len(categories)  # Number of classes (should be 3)

    model = TextClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []
    val_losses = []
    val_accuracies = []  # Track validation accuracy

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
        model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                total_val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == batch_y).sum().item()
                total += batch_y.size(0)

        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / total  # Compute validation accuracy
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")

    # Plot training & validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Evaluate on Test Set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
            total += batch_y.size(0)

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return model, test_loader

def get_predictions(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1)  # Get class with highest probability
            predictions.extend((predicted - 1).cpu().numpy())  # Convert to list
    return predictions

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

def save_txt_data(filename, data):
    np.savetxt(filename, data, fmt="%.18e")  # Save numerical data with six decimal places

# Function to save labels
def save_txt_labels(filename, labels):
    np.savetxt(filename, labels, fmt="%d")  # Save labels as integers
    
def save_labels_and_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()

    # Save feature matrices
    save_txt_data("Data/X_train.txt", X_train_dense)
    save_txt_data("Data/X_val.txt", X_val_dense)
    save_txt_data("Data/X_test.txt", X_test_dense)

    # Save labels
    save_txt_labels("Data/y_train.txt", y_train)
    save_txt_labels("Data/y_val.txt", y_val)
    save_txt_labels("Data/y_test.txt", y_test)
    
def verify_data_integrity(original, filename):
    # Reload the saved data
    reloaded = np.loadtxt(filename)

    # Check if all values are almost equal (accounting for minor floating-point differences)
    if np.allclose(original, reloaded, atol=0):  # Tolerance set to 1e-10 for high precision
        print(f" Verification passed: {filename} data is intact!")
    else:
        print(f" Verification failed: {filename} data has discrepancies!")

def main():
    # test new text
    test_texts = [
        "I'm thrilled with the recent performance of $AAPL! The new product launch exceeded expectations.",
        "The market is unpredictable today. Not sure how $GOOG will move.",
        "Disappointed with $TSLA's quarterly results. The numbers were below projections."
    ]
    df = pd.read_csv("Step3/FinalData/combined.csv")
    texts = df['body']
    labels = df['label']
    categories = ["negative", "neutral", "positive"]
    
    pre_processed_text = lemmatization(texts)
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        min_df=5,
        preprocessor=lambda x: x,  # keep text unchanged, already preprocessed
        tokenizer=lambda x: x.split()  # simple tokenization, text already standardized
    )
    X_tfidf = tfidf_vectorizer.fit_transform(pre_processed_text)
    
    
    # X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42)
    # X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42) 
    
    # Currently in Data Folder
    X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    
    # model = logisticRegressionModel(X_train, y_train, X_test)

    # print("\nTesting new texts:")
    # for text in test_texts:
    #     result = predict_category_for_logistic_regression(text, model, tfidf_vectorizer, categories)
    #     print(f"\nText: {text}")
    #     print(f"Category: {result['category']}")
    #     print(f"Confidence: {result['confidence']:.2f}")
    #     print("-" * 50)
        
    # output_results(model, X_train, y_train, X_val,y_val, X_test, y_test, categories)
    
    
    model, test_loader = neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories)
    
    predicted_labels = get_predictions(model, test_loader)
    save_txt_labels("Data/y_predicted.txt", predicted_labels)
    # # print(predicted_labels)
    
    # print("\nTesting new texts:")
    # for text in test_texts:
    #     result = predict_category_for_neural_network(text, model, tfidf_vectorizer, categories)
    #     print(f"\nText: {text}")
    #     print(f"Category: {result['category']}")
    #     print(f"Confidence: {result['confidence']:.2f}")
    
    
if __name__ == "__main__":
    main()