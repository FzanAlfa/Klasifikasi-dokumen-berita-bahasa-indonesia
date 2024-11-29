import torch
import pandas as pd
import fasttext
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import torch.nn as nn

# Konfigurasi utama
FASTTEXT_EMBEDDING_DIM = 300
NUM_CLASSES = 7

# Dataset custom untuk pengujian
class TestDataset(Dataset):
    def __init__(self, csv_file, fasttext_model, label_to_idx):
        self.data = []
        self.labels = []
        self.fasttext_model = fasttext_model
        self.label_to_idx = label_to_idx

        # Membaca data dari CSV
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            self.data.append(row["isi"])
            self.labels.append(label_to_idx[row["label"]])

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
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = torch.max(x, dim=2).values  # Global max pooling
        x = self.fc(x)
        return x

# Fungsi untuk pengujian
def test_model(model, dataloader, idx_to_label):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for texts, labels in dataloader:
            texts, labels = texts.to(device), labels.to(device)

            outputs = model(texts)
            _, preds = torch.max(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

            # Cetak hasil per teks
            for i in range(len(labels)):
                print(f"Text: {texts[i].size()}")  # Display padded size
                print(f"Label: {idx_to_label[labels[i].item()]}, Predicted: {idx_to_label[preds[i].item()]}")

    # Hitung akurasi
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")

# Main program
if __name__ == "__main__":
    # Load FastText model
    print("Loading FastText model...")
    fasttext_model = fasttext.load_model("cc.id.300.bin")  # Ganti dengan model FastText bahasa Indonesia

    # Label mapping
    label_to_idx = {
        "makanan": 0,
        "ekonomi": 1,
        "gosip": 2,
        "internet": 3,
        "olahraga": 4,
        "otomotif": 5,
        "dunia": 6
    }
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Load dataset pengujian
    test_dataset = TestDataset("inputtestingisi.csv", fasttext_model, label_to_idx)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    # Load model CNN
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNClassifier(embedding_dim=FASTTEXT_EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load("cnn_classifier.pth"))
    print("Model loaded.")

    # Uji model
    test_model(model, test_dataloader, idx_to_label)
