import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from aim import Run, Image
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Общие параметры
batch_size = 64
epochs = 2
lr = 0.001
train_subset_size = 5000
val_subset_size = 1000

# Датасет
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
full_train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
full_val_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform)

train_dataset = Subset(full_train_dataset, range(train_subset_size))
val_dataset = Subset(full_val_dataset, range(val_subset_size))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Модель 1: SimpleNet (MLP)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Модель 2: ConvNet (CNN)
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 7 * 7)
        return self.fc(x)

# Обучение + логгирование модели
def train_and_log(model, model_name, lr):
    run = Run(experiment="FashionMNIST_Model_Comparison")
    run["model_name"] = model_name
    run["hparams"] = {"lr": lr, "batch_size": batch_size, "epochs": epochs, "model": model_name}

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss / len(train_loader)
        run.track(train_loss, name="train_loss", step=epoch)
        run.track(train_acc, name="train_accuracy", step=epoch)

        # Валидация
        model.eval()
        val_correct, val_total = 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = val_correct / val_total
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        run.track(val_acc, name="val_accuracy", step=epoch)
        run.track(val_f1, name="val_f1_score", step=epoch)

        # Confusion matrix
        cm = confusion_matrix(val_labels, val_preds)
        disp = ConfusionMatrixDisplay(cm)
        fig, ax = plt.subplots(figsize=(5, 5))
        disp.plot(ax=ax)
        ax.set_title(f"{model_name} - Confusion Matrix (Epoch {epoch})")

        run.track(Image(fig), name="confusion_matrix", step=epoch)
        plt.close(fig)

# Запускаем эксперименты
train_and_log(SimpleNet(), "SimpleNet_MLP", 0.001)
train_and_log(SimpleNet(), "SimpleNet_MLP", 0.0001)
train_and_log(ConvNet(), "ConvNet_CNN", 0.001)
