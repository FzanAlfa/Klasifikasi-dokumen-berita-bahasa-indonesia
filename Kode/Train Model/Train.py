import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import fasttext

# Konfigurasi utama
FASTTEXT_EMBEDDING_DIM = 300
NUM_CLASSES = 7
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.0001

# Mapping label manual
label_to_idx = {
    "makanan": 0,
    "ekonomi": 1,
    "gosip": 2,
    "internet": 3,
    "olahraga": 4,
    "otomotif": 5,
    "dunia": 6
}
idx_to_label = {v: k for k, v in label_to_idx.items()}

# Dataset custom
class NewsDataset(Dataset):
    def __init__(self, file_path, fasttext_model, label_to_idx):
        self.data = []
        self.labels = []
        self.fasttext_model = fasttext_model
        self.label_to_idx = label_to_idx

        # Membaca data dari file dan memisahkan label serta teks
        with open(file_path, 'r') as f:
            for line in f:
                label, text = line.split(maxsplit=1)
                self.data.append(text.strip())
                self.labels.append(self.label_to_idx[label.replace('__label__', '')])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]

        # Embedding menggunakan FastText
        words = text.split()
        embeddings = [self.fasttext_model.get_word_vector(word) for word in words]
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        return embeddings, label

# Collate function untuk padding sequence
def collate_fn(batch):
    texts, labels = zip(*batch)

    # Padding sequence agar memiliki panjang yang sama
    max_len = max(text.size(0) for text in texts)
    padded_texts = torch.zeros(len(texts), max_len, FASTTEXT_EMBEDDING_DIM)

    for i, text in enumerate(texts):
        padded_texts[i, :text.size(0), :] = text

    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels

# Model CNN
class CNNClassifier(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(CNNClassifier, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Mengubah dimensi agar cocok dengan Conv1d (batch_size, embedding_dim, seq_len)
        x = x.permute(0, 2, 1)

        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = torch.max(x, dim=2).values  # Global max pooling
        x = self.fc(x)
        return x

# Fungsi untuk melatih model
def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Fungsi untuk menguji model
def test_model(model, dataloader, idx_to_label):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")

# Main program
if __name__ == "__main__":
    # Load FastText model pre-trained (skipgram atau cbow)
    print("Loading FastText model...")
    fasttext_model = fasttext.load_model("cc.id.300.bin")  # Ganti dengan model FastText bahasa Indonesia

    # Load dataset
    dataset = NewsDataset("dataset1.txt", fasttext_model, label_to_idx)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)

    # Model, loss function, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(embedding_dim=FASTTEXT_EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    train_model(model, dataloader, criterion, optimizer, EPOCHS)

    # Save model
    torch.save(model.state_dict(), "cnn_classifier.pth")
    print("Model saved to cnn_classifier.pth")
