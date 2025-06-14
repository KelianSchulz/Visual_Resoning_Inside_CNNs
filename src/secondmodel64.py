from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import os
import torchvision.transforms as transforms
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from torch.utils.data import WeightedRandomSampler
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
import matplotlib.pyplot as plt




device = t.device("cuda" if t.cuda.is_available() else "cpu")
batch_size = 4



# --------------------- Dataset ---------------------
class ConceptDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(self.data['concept_label'].unique()))}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label_str = self.data.iloc[idx]['concept_label']
        label = self.label_to_idx[label_str]
        
        image = Image.open(img_path).convert("L")
        if self.transform:
            image = self.transform(image)
        
        return image, label

# --------------------- Transformation ---------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1), 
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# --------------------- Data Split ---------------------
full_df = pd.read_csv("./data/concept_labels.csv")
train_df, val_df = train_test_split(full_df, test_size=0.3, stratify=full_df["concept_label"], random_state=42)

train_dataset = ConceptDataset(train_df, transform=transform)
val_dataset = ConceptDataset(val_df, transform=val_transform)


counts = np.array([400, 200, 300, 300, 1000, 400, 1000, 300, 300, 1300])

# --------------------- Class Balancing Setup ---------------------
class_weights_sampler = 1.0 / counts
class_weights_loss = t.tensor(class_weights_sampler, dtype=t.float32, device=device)

sample_weights = class_weights_sampler[train_df['concept_label'].map(train_dataset.label_to_idx).values]
sample_weights = t.tensor(sample_weights, dtype=t.float32)

sampler = WeightedRandomSampler(
    weights=sample_weights.tolist(),
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#---------------------Model---------------------
class ConceptCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
 
        self.dropout = nn.Dropout(0.05)  # hilft gegen Overfitting
        self.fc1 = nn.Linear(256 * 8 * 8, 512) 
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))   # 64x64 → 32x32
        x = self.pool(F.relu(self.bn2(self.conv2(x))))   # 32x32 → 16x16
        x = self.pool(F.relu(self.bn3(self.conv3(x))))   # 16x16 → 8x8
        x = x.view(-1, 256 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
# --------------------- Loss ------------------
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, inputs, targets):
        logp = F.log_softmax(inputs, dim=1)
        p = t.exp(logp)
        loss = F.nll_loss((1 - p)**self.gamma * logp, targets, weight=self.weight)
        return loss
    
model = ConceptCNN().to(device)
criterion = FocalLoss(gamma=2, weight=class_weights_loss)
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6 )



# --------------------- Training + Validation ---------------------

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

def train(num_epochs=200, patience=10):
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.7,
        patience=3
    )
   
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        # --------- Training ------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        epoch_loss = 0
        for i, (inputs, labels) in enumerate(train_loader, 1):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            if i % 10 == 0:
                # Durchschnitt der letzten 10 Batches
                batch_avg_loss = running_loss / 10
                batch_acc = 100 * correct_train / total_train
                print(f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Batch {i}: Train Loss {batch_avg_loss:.4f} | Train Acc {batch_acc:.2f}%")
                running_loss = 0.0  # reset für die nächsten 10 Batches

        # Nach dem kompletten Epoch-Loop:
        avg_train_loss = epoch_loss / len(train_loader)  # Mittel aller Batches
        avg_train_acc  = 100 * correct_train / total_train
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)

            
        # --------- Validation ------------
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with t.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        
        
        

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_acc = 100 * correct_val / total_val
        val_accuracies.append(val_acc)
        print(f">>> Epoch {epoch+1} Summary: Val Loss {avg_val_loss:.4f} | Val Acc {val_acc:.2f}%")

        # --------- Scheduler & Early Stopping ------------
        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Besten Checkpoint speichern
            t.save(model.state_dict(), "./data/best_concept_net.pth")
            print("    (✔️ New best model saved)")
        else:
            epochs_no_improve += 1
            print(f"    (⚠️ No improvement for {epochs_no_improve} epochs)")

        if epochs_no_improve >= patience:
            print(f"Early stopping after {epoch+1} epochs.")
            break

    print("Training beendet. Beste Val Loss:", best_val_loss)
   
def test():
    model.load_state_dict(t.load("./data/best_concept_net.pth"))
    model.to(device)
    model.eval()
    print("Modell geladen und in Eval-Modus gesetzt.")

    correct = 0
    total = 0
    batch_count = 0

    with t.no_grad():
        for data in val_loader:
            batch_count += 1
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Genauigkeit auf dem Validierungsset: {accuracy:.2f}%")
    print(f"Anzahl der Validierungs-Batches: {batch_count}")

def visualize_random_prediction(model, val_df, transform, label_to_idx):
    # Umkehrung des Label-Mappings
    idx_to_label = {v: k for k, v in label_to_idx.items()}

    # Zufällige Zeile aus dem DataFrame auswählen
    row = val_df.sample(1).iloc[0]
    img_path = row["filepath"]
    true_label = row["concept_label"]

    # Bild laden und transformieren
    image = Image.open(img_path).convert("L")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Modell in den Eval-Modus versetzen und vorhersagen
    model.eval()
    with t.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = idx_to_label[pred_idx]

    # Bild anzeigen
    plt.imshow(image)
    plt.axis("off")
    plt.title(f"Wahr: {true_label} | Vorhergesagt: {pred_label}")
    plt.show()





if __name__ == "__main__":
    train()
    test()

    # Jetzt sind die Listen gefüllt – plotten:
    import matplotlib.pyplot as plt

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(10,4))

    # Loss
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses,   label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy
    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Train Acc')
    plt.plot(epochs, val_accuracies,   label='Val   Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

# visualize_random_prediction(
#     model,
#     val_df,
#     transform,
#     train_dataset.label_to_idx
# )