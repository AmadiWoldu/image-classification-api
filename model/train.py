# =====================================
# 1. Importing Libraries
# =====================================
import os
import wandb
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from tqdm.auto import tqdm 

# =====================================
# 2. Setting Random Seeds & Device
# =====================================
torch.manual_seed(42) 
torch.cuda.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu' 

# =====================================
# 3. Initializing WandB
# =====================================
wandb.init(
    project = 'image-classificaton-api',
    name = 'ResNet18-model'
)

# =====================================
# 4. Defining Data Transforms
# =====================================
train_transform = transforms.Compose([
    # transforms.Resize(size = (224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    # transforms.Resize(size = (224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# =====================================
# 5. Loading Datasets
# =====================================
trainset = torchvision.datasets.CIFAR10(
    root = 'data',
    train = True,
    transform=train_transform,
    download=False
)
testset = torchvision.datasets.CIFAR10(
    root = 'data',
    train = False,
    transform = test_transform,
    download=False
)

# =====================================
# 6. Creating Data Loaders
# =====================================
train_dataloader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = 256,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

test_dataloader = torch.utils.data.DataLoader(
    dataset = testset,
    shuffle=False,
    num_workers=0,
    batch_size=256,
    pin_memory=True
)

# =====================================
# 7. Building the Model
# =====================================
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    in_features=model.fc.in_features,
    out_features=10
)

# =====================================
# 8. Freezing Layers for Fine-Tuning
# =====================================
# All layers unfrozen — full fine-tuning
for param in model.parameters():
    param.requires_grad = True

# =====================================
# 9. Defining Training Step
# =====================================
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    model = model.to(device=device)
    model.train()

    train_loss, train_acc = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    train_loss /= len(dataloader)
    train_acc /= len(dataloader)
    return train_loss, train_acc

# =====================================
# 10. Defining Testing Step
# =====================================
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model = model.to(device=device)
    model.eval()

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            test_pred_logits = model(X)

            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += (test_pred_labels == y).sum().item() / len(test_pred_labels)

        test_loss /= len(dataloader)
        test_acc /= len(dataloader)
        return test_loss, test_acc

# =====================================
# 11. Defining Full Training Loop
# =====================================
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    model = model.to(device=device)
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Log per-epoch metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_loss": test_loss,
            "test_acc": test_acc
        })

        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    return results

# =====================================
# 12. Setting Hyperparameters
# =====================================
NUM_EPOCHS = 10
loss_fn = nn.CrossEntropyLoss()
# FIX: use model.parameters() to optimize ALL layers, not just FC
optimizer = torch.optim.Adam([
    {"params": model.fc.parameters(),          "lr": 0.001},   # FC head — freshly initialized
    {"params": [p for name, p in model.named_parameters()
                if "fc" not in name],           "lr": 0.00001}  # Backbone — pretrained, be gentle
])

# =====================================
# 13. Running Training & Timing
# =====================================
from timeit import default_timer as timer
start_time = timer()

model_results = train(model=model,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=NUM_EPOCHS)

end_time = timer()
train_time = end_time - start_time
print(f"Total training time: {end_time-start_time:.3f} seconds")

# =====================================
# 14. Evaluating on Test Set
# =====================================
Loss, Accuracy = test_step(model=model, dataloader=test_dataloader, loss_fn=loss_fn)
print(f"Test Loss: {Loss:.4f} | Test Accuracy: {Accuracy:.4f}")

# =====================================
# 15. Logging Metrics to WandB
# =====================================
wandb.config.update({
    "model": "ResNet18",
    "epochs": NUM_EPOCHS,
    "batch_size": 256,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "fine_tuning": "full"
})

wandb.log({
    "Loss": Loss,
    "Accuracy": Accuracy,
    "train_time": train_time
})

# =====================================
# 16. Saving Model Checkpoint
# =====================================
save_path = "model/artifacts/model_state_dict.pth"

torch.save({
    "model_state_dict": model.state_dict(),
    "class_names": trainset.classes
}, save_path)

# =====================================
# 17. Logging Model Artifact to WandB
# =====================================
artifact = wandb.Artifact(
    name="ResNet18-v1",
    type="model",
    description="ResNet18 fully fine-tuned on CIFAR-10 for 10 epochs",
    metadata={
        "accuracy": Accuracy,
        "loss": Loss,
        "framework": "pytorch",
        "epochs": NUM_EPOCHS
    }
)

artifact.add_file(save_path)
wandb.log_artifact(artifact)

# =====================================
# 18. Finishing WandB Run
# =====================================
wandb.finish()