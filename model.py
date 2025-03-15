import spacy
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

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

def neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories):
    X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val.toarray(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)

    y_train_tensor = torch.tensor((y_train+1).values, dtype=torch.long)  
    y_val_tensor = torch.tensor((y_val+1).values, dtype=torch.long)
    y_test_tensor = torch.tensor((y_test+1).values, dtype=torch.long)

    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1] 
    hidden_dim = 128
    output_dim = len(categories) 

    model = TextClassifier(input_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 5
    train_losses = []
    val_losses = []
    val_accuracies = [] 

    for epoch in range(num_epochs):
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
        val_accuracy = correct / total 
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}")

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

    test_accuracy = correct / total
    print(f"Test Accuracy: {test_accuracy:.2f}")

    return model, test_loader

def get_predictions(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs, 1) 
            predictions.extend((predicted - 1).cpu().numpy()) 
    return predictions

def save_txt_data(filename, data):
    np.savetxt(filename, data, fmt="%.18e")

def save_txt_labels(filename, labels):
    np.savetxt(filename, labels, fmt="%d")
    
def save_labels_and_data(X_train,X_val,X_test,y_train,y_val,y_test):
    X_train_dense = X_train.toarray()
    X_val_dense = X_val.toarray()
    X_test_dense = X_test.toarray()

    save_txt_data("RandomData/X_train.txt", X_train_dense)
    save_txt_data("RandomData/X_val.txt", X_val_dense)
    save_txt_data("RandomData/X_test.txt", X_test_dense)

    save_txt_labels("RandomData/y_train.txt", y_train)
    save_txt_labels("RandomData/y_val.txt", y_val)
    save_txt_labels("RandomData/y_test.txt", y_test)
    
def verify_data_integrity(original, filename):
    reloaded = np.loadtxt(filename)

    if np.allclose(original, reloaded, atol=0):  
        print(f" Verification passed: {filename} data is intact!")
    else:
        print(f" Verification failed: {filename} data has discrepancies!")

def main():
    df = pd.read_csv("Step3/FinalData/combined.csv")
    texts = df['body']
    labels = df['label']
    categories = ["negative", "neutral", "positive"]
    
    pre_processed_text = lemmatization(texts)
    
    tfidf_vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        min_df=5,
        preprocessor=lambda x: x,
        tokenizer=lambda x: x.split() 
    )
    X_tfidf = tfidf_vectorizer.fit_transform(pre_processed_text)
        
    X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, labels, test_size=0.3, random_state=42, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    model, test_loader = neural_network(X_train, X_test, X_val, y_train, y_val, y_test, categories)
    
    # predicted_labels = get_predictions(model, test_loader)
    # save_labels_and_data(X_train, X_val, X_test, y_train, y_val, y_test)
    # save_txt_labels("RandomData/y_predicted.txt", predicted_labels)
    
if __name__ == "__main__":
    main()