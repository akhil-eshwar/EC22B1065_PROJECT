import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix

from dataset import LiTSDataset
from unetpp_model import UNetPP


# ---------------------------------------------------
# Metrics
# ---------------------------------------------------
def compute_metrics(cm):

    total = cm.sum()

    precision_list = []
    recall_list = []
    specificity_list = []
    dice_list = []
    iou_list = []

    for cls in range(1,3):

        TP = cm[cls,cls]
        FP = cm[:,cls].sum() - TP
        FN = cm[cls,:].sum() - TP
        TN = total - TP - FP - FN

        precision = TP/(TP+FP+1e-8)
        recall = TP/(TP+FN+1e-8)
        specificity = TN/(TN+FP+1e-8)

        dice = 2*TP/(2*TP+FP+FN+1e-8)
        iou = TP/(TP+FP+FN+1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        dice_list.append(dice)
        iou_list.append(iou)

    return {
        "dice": np.mean(dice_list),
        "iou": np.mean(iou_list),
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "specificity": np.mean(specificity_list)
    }


# ---------------------------------------------------
# Main
# ---------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = LiTSDataset(
    r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\2D_Slices\images",
    r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\2D_Slices\masks"
)

val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1)

model = UNetPP(1,3).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_dice = 0

for epoch in range(15):

    print(f"\nEpoch {epoch+1}/15")

    # ---------------- TRAIN ----------------
    model.train()
    train_loss = 0

    for images, masks in tqdm(train_loader):

        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---------------- VALIDATION ----------------
    model.eval()

    cm_total = np.zeros((3,3), dtype=np.int64)

    with torch.no_grad():

        for images, masks in val_loader:

            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)

            preds = torch.argmax(outputs, dim=1)

            cm = confusion_matrix(
                masks.view(-1).cpu().numpy(),
                preds.view(-1).cpu().numpy(),
                labels=[0,1,2]
            )

            cm_total += cm

    metrics = compute_metrics(cm_total)

    print("Dice:", round(metrics["dice"],4))
    print("IoU:", round(metrics["iou"],4))
    print("Precision:", round(metrics["precision"],4))
    print("Recall:", round(metrics["recall"],4))
    print("Specificity:", round(metrics["specificity"],4))

    if metrics["dice"] > best_dice:

        best_dice = metrics["dice"]

        torch.save(
            model.state_dict(),
            "best_unetpp.pth"
        )

        print("Best model saved")