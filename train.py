import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from dataset import LiTSDataset
from model import UNet


# ---------------------------
# Metric Function
# ---------------------------

def compute_metrics(cm):

    metrics = {}
    total = cm.sum()
    accuracy = np.trace(cm) / total

    precision_list = []
    recall_list = []
    specificity_list = []
    f1_list = []
    dice_list = []
    iou_list = []

    num_classes = cm.shape[0]

    for cls in range(1, num_classes):

        TP = cm[cls, cls]
        FP = cm[:, cls].sum() - TP
        FN = cm[cls, :].sum() - TP
        TN = total - TP - FP - FN

        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        f1 = 2 * TP / (2 * TP + FP + FN + 1e-8)
        dice = f1
        iou = TP / (TP + FP + FN + 1e-8)

        precision_list.append(precision)
        recall_list.append(recall)
        specificity_list.append(specificity)
        f1_list.append(f1)
        dice_list.append(dice)
        iou_list.append(iou)

    metrics["accuracy"] = accuracy
    metrics["precision"] = np.mean(precision_list)
    metrics["recall"] = np.mean(recall_list)
    metrics["specificity"] = np.mean(specificity_list)
    metrics["f1"] = np.mean(f1_list)
    metrics["dice"] = np.mean(dice_list)
    metrics["iou"] = np.mean(iou_list)

    return metrics


# ---------------------------
# Training
# ---------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--images_dir", type=str,
                        default=r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\2D_Slices\images")

    parser.add_argument("--masks_dir", type=str,
                        default=r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\2D_Slices\masks")

    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    torch.backends.cudnn.benchmark = True

    # ---------------- Dataset ----------------

    dataset = LiTSDataset(args.images_dir, args.masks_dir)

    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print("Train samples:", train_size)
    print("Val samples:", val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,      # safer on Windows
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # ---------------- Model ----------------

    model = UNet(n_channels=1, n_classes=3).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scaler = torch.amp.GradScaler("cuda")

    best_dice = 0

    print("\nTraining started\n")

    # ---------------- Epoch Loop ----------------

    for epoch in range(args.epochs):

        print(f"\n===== Epoch {epoch+1}/{args.epochs} =====")
        epoch_start = time.time()

        # ---------- TRAIN ----------

        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc="Training"):

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):

                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # ---------- VALIDATION ----------

        torch.cuda.empty_cache()

        model.eval()

        val_loss = 0
        cm_total = np.zeros((3, 3), dtype=np.int64)

        with torch.no_grad():

            for images, masks in tqdm(val_loader, desc="Validation"):

                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                cm = confusion_matrix(
                    masks.view(-1).cpu().numpy(),
                    preds.view(-1).cpu().numpy(),
                    labels=[0,1,2]
                )

                cm_total += cm

                del images, masks, outputs, preds

        avg_val_loss = val_loss / len(val_loader)

        metrics = compute_metrics(cm_total)

        print("\nResults:")
        print("Train Loss:", avg_train_loss)
        print("Val Loss:", avg_val_loss)
        print("Dice:", metrics["dice"])
        print("IoU:", metrics["iou"])

        if metrics["dice"] > best_dice:

            best_dice = metrics["dice"]
            torch.save(model.state_dict(), "best_unet.pth")
            print("Saved best model")

        print("Epoch time:", time.time() - epoch_start)

        # ---------- Visualization ----------

        images, masks = next(iter(val_loader))
        images = images.to(device)

        with torch.no_grad():
            preds = torch.argmax(model(images), dim=1)

        img = images[0].cpu().squeeze()
        gt = masks[0]
        pr = preds[0].cpu()

        plt.figure(figsize=(12,4))

        plt.subplot(1,3,1)
        plt.title("CT")
        plt.imshow(img, cmap="gray")

        plt.subplot(1,3,2)
        plt.title("GT")
        plt.imshow(gt)

        plt.subplot(1,3,3)
        plt.title("Prediction")
        plt.imshow(pr)

        plt.show()

    print("\nTraining complete")


if __name__ == "__main__":
    main()
