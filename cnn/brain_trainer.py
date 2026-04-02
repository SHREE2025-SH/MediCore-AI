"""
MediCore AI — Brain MRI Tumor CNN Trainer
ResNet50 Transfer Learning | PyTorch
Classes: glioma / meningioma / notumor / pituitary
"""
import os, copy, time, json, torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR   = Path(r"D:\MediCore_AI")
TRAIN_DIR  = BASE_DIR / "cnn" / "brain" / "Training"
TEST_DIR   = BASE_DIR / "cnn" / "brain" / "Testing"
MODEL_DIR  = BASE_DIR / "cnn"
MODEL_PATH = MODEL_DIR / "brain_resnet50.pth"
META_PATH  = MODEL_DIR / "brain_meta.json"

IMG_SIZE    = 224
BATCH_SIZE  = 128
EPOCHS      = 20
LR          = 1e-4
PATIENCE    = 5
NUM_WORKERS = 4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

print(f"[Brain CNN] Device: {DEVICE}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def get_loaders():
    full_train = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    val_size   = int(0.15 * len(full_train))
    train_size = len(full_train) - val_size
    train_ds, val_ds = random_split(full_train, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = val_transform
    test_ds = datasets.ImageFolder(TEST_DIR, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = full_train.classes
    print(f"[Data] Classes : {class_names}")
    print(f"[Data] Train   : {train_size} | Val: {val_size} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader, class_names

def build_model(num_classes):
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    for param in model.parameters(): param.requires_grad = False
    for param in model.layer4.parameters(): param.requires_grad = True
    in_f = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_f, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model.to(DEVICE)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = criterion(out, y)
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total

def unfreeze_backbone(model):
    for param in model.parameters(): param.requires_grad = True
    return optim.Adam([
        {"params": model.layer1.parameters(), "lr": 1e-6},
        {"params": model.layer2.parameters(), "lr": 3e-6},
        {"params": model.layer3.parameters(), "lr": 5e-6},
        {"params": model.layer4.parameters(), "lr": 1e-5},
        {"params": model.fc.parameters(),     "lr": 2e-5},
    ])

def main():
    train_loader, val_loader, test_loader, class_names = get_loaders()
    num_classes = len(class_names)
    model = build_model(num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_acc, best_weights, patience_ctr = 0.0, None, 0
    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

    print("\n[Phase 1] Training head...")
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler.step(vl_loss)
        for k,v in zip(["train_loss","train_acc","val_loss","val_acc"],[tr_loss,tr_acc,vl_loss,vl_acc]):
            history[k].append(v)
        print(f"  Epoch {epoch:02d}/{EPOCHS} | Train: {tr_acc:.4f} | Val: {vl_acc:.4f} | {time.time()-t0:.1f}s")
        if vl_acc > best_acc:
            best_acc = vl_acc; best_weights = copy.deepcopy(model.state_dict()); patience_ctr = 0
            print(f"  ✅ Best val acc: {best_acc:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE: print(f"  ⏹ Early stop"); break

    print("\n[Phase 2] Fine-tuning backbone...")
    model.load_state_dict(best_weights)
    optimizer2 = unfreeze_backbone(model)
    scheduler2 = ReduceLROnPlateau(optimizer2, mode="min", factor=0.5, patience=2)
    patience_ctr2 = 0

    for epoch in range(1, 11):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer2)
        vl_loss, vl_acc = evaluate(model, val_loader, criterion)
        scheduler2.step(vl_loss)
        for k,v in zip(["train_loss","train_acc","val_loss","val_acc"],[tr_loss,tr_acc,vl_loss,vl_acc]):
            history[k].append(v)
        print(f"  Fine {epoch:02d}/10 | Train: {tr_acc:.4f} | Val: {vl_acc:.4f} | {time.time()-t0:.1f}s")
        if vl_acc > best_acc:
            best_acc = vl_acc; best_weights = copy.deepcopy(model.state_dict()); patience_ctr2 = 0
            print(f"  ✅ Best val acc: {best_acc:.4f}")
        else:
            patience_ctr2 += 1
            if patience_ctr2 >= PATIENCE: print(f"  ⏹ Early stop"); break

    model.load_state_dict(best_weights)
    torch.save({"model_state_dict": best_weights, "class_names": class_names,
                "num_classes": num_classes, "img_size": IMG_SIZE}, MODEL_PATH)
    te_loss, te_acc = evaluate(model, test_loader, criterion)
    print(f"\n[Test] Acc={te_acc:.4f}")
    json.dump({"model_type":"resnet50","task":"brain_tumor","class_names":class_names,
               "num_classes":num_classes,"img_size":IMG_SIZE,"mean":MEAN,"std":STD,
               "best_val_acc":round(best_acc,4),"test_acc":round(te_acc,4)},
              open(META_PATH,"w"), indent=2)
    print(f"[Save] {MODEL_PATH}\n[Brain CNN] Done! 🎉 Best Val: {best_acc:.4f} | Test: {te_acc:.4f}")

if __name__ == "__main__":
    main()