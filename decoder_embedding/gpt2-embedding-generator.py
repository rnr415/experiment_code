import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2Model
import numpy as np
from pytorch_metric_learning import losses, miners

class SentenceDataset(Dataset):
    def __init__(self, sentences, labels):
        self.sentences = sentences
        self.labels = labels

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.labels[idx]

def generate_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    embedding = torch.mean(last_hidden_states, dim=1)
    return embedding.squeeze()

class EmbeddingModel(nn.Module):
    def __init__(self, model_name='gpt2'):
        super(EmbeddingModel, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)

    def forward(self, text):
        return generate_embedding(text, self.tokenizer, self.model)

def train(model, train_loader, optimizer, loss_func, miner, device, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_sentences, batch_labels in train_loader:
            optimizer.zero_grad()
            embeddings = torch.stack([model(sentence) for sentence in batch_sentences]).to(device)
            
            # Use the miner to find hard pairs
            hard_pairs = miner(embeddings, batch_labels.to(device))
            
            # Compute the loss using the mined pairs
            loss = loss_func(embeddings, batch_labels.to(device), hard_pairs)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Example usage
sentences = [
    "The cat sits on the mat.",
    "A feline rests on the rug.",
    "Dogs are man's best friend.",
    "Canines are loyal companions.",
    "The sun rises in the east.",
    "Daybreak occurs in the morning."
]
labels = torch.tensor([0, 0, 1, 1, 2, 2])  # Similar sentences have the same label

dataset = SentenceDataset(sentences, labels)
train_loader = DataLoader(dataset, batch_size=6, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmbeddingModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Initialize the InfoNCE loss and miner
loss_func = losses.NTXentLoss(temperature=0.07)
miner = miners.MultiSimilarityMiner()

train(model, train_loader, optimizer, loss_func, miner, device)

# Generate embeddings for new sentences
new_sentences = [
    "A kitten is sleeping on the carpet.",
    "The loyal dog follows its owner.",
    "The moon rises at night."
]

model.eval()
with torch.no_grad():
    new_embeddings = torch.stack([model(sentence) for sentence in new_sentences])

print("New sentence embeddings:")
print(new_embeddings)
