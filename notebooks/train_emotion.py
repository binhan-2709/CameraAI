"""
Script to train a custom facial expression recognition (FER) model.
Supports loading and merging multiple datasets: FER-2013, FERPlus, CK+, and JAFFE.
Exports the trained model to ONNX format for deployment.

Usage:
    python notebooks/train_emotion.py --epochs 10 --batch_size 64
"""

import os
import argparse
import time
from pathlib import Path
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("[Warning] PyTorch is not installed in the environment. Please run:")
    print("  pip install torch torchvision")
    # We will define stubs so that the script can compile and run basic help/argument parser
    torch = None
    nn = None
    Dataset = object

# Unified emotion classes (mapping FERPlus classes)
EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "contempt", "fear"
]


BaseModule = nn.Module if torch is not None else object


class MiniXception(BaseModule):
    """
    A lightweight Convolutional Neural Network (CNN) architecture inspired by Xception.
    Extremely fast to train and run, fits within ~5MB, ideal for real-time CPU deployment.
    """
    def __init__(self, num_classes=8):
        super(MiniXception, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        
        # Residual Blocks
        self.residual_block1 = self._make_residual_block(8, 16)
        self.residual_block2 = self._make_residual_block(16, 32)
        self.residual_block3 = self._make_residual_block(32, 64)
        self.residual_block4 = self._make_residual_block(64, 128)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2),
            # Shortcut mapping to match dimensions
        )

    def forward(self, x):
        # Initial layers
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Residual pathways (simplified for size)
        # Block 1
        res1 = self.residual_block1(x)
        # Block 2
        res2 = self.residual_block2(res1)
        # Block 3
        res3 = self.residual_block3(res2)
        # Block 4
        res4 = self.residual_block4(res3)
        
        x = self.global_pool(res4)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class EmotionTrainingDataset(Dataset):
    """
    A unified Dataset wrapper that supports raw images or a synthetic mock mode.
    Can load FER-2013 CSV, CK+ images, or JAFFE directories.
    """
    def __init__(self, data_list, img_size=64, is_train=True):
        """
        Args:
            data_list: list of tuples (img_path_or_array, label_idx)
            img_size: target image width/height (typically 64)
            is_train: apply augmentations if True
        """
        self.data_list = data_list
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item, label = self.data_list[idx]
        
        # Load image
        if isinstance(item, np.ndarray):
            img = item
        else:
            img = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)

        # Resize to standard model size
        img = cv2.resize(img, (self.img_size, self.img_size))

        # Basic online augmentations if training
        if self.is_train:
            # Horizontal flip
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
            # Random slight rotation
            if np.random.rand() > 0.5:
                angle = np.random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D((self.img_size//2, self.img_size//2), angle, 1.0)
                img = cv2.warpAffine(img, M, (self.img_size, self.img_size))

        # Normalize and add channel dimension
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0) # Shape: (1, H, W)
        
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


def generate_mock_data(num_samples=1000):
    """Generates synthetic face-like structures for mock/testing training pipeline."""
    data = []
    print(f"[Dataset] Generating {num_samples} mock face images to test pipeline...")
    for _ in range(num_samples):
        # Create a mock grayscale canvas
        img = np.zeros((64, 64), dtype=np.uint8) + np.random.randint(20, 100)
        # Draw a mock face circle
        cv2.circle(img, (32, 32), np.random.randint(18, 26), np.random.randint(150, 220), -1)
        # Draw mock eyes
        cv2.circle(img, (22, 24), 3, 20, -1)
        cv2.circle(img, (42, 24), 3, 20, -1)
        
        # Random emotion label mapping
        label = np.random.randint(0, 8)
        # Draw mouth based on label
        if label == 1: # Happiness: smile
            cv2.ellipse(img, (32, 42), (10, 5), 0, 0, 180, 20, 2)
        elif label in (3, 4): # Sadness / Anger: frown
            cv2.ellipse(img, (32, 46), (8, 4), 0, 180, 360, 20, 2)
        else: # Neutral / Surprise: flat line or circle
            cv2.line(img, (24, 44), (40, 44), 20, 2)
            
        # Add random noise
        noise = np.random.normal(0, 5, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        data.append((img, label))
    return data


def scan_datasets(dataset_paths):
    """Scans paths to gather files from FER-2013, CK+, JAFFE, etc."""
    combined_data = []
    
    # 1. Look for CK+ dataset
    ck_path = dataset_paths.get("ck")
    if ck_path and os.path.exists(ck_path):
        print(f"[Dataset] Scanning CK+ in: {ck_path}")
        p = Path(ck_path)
        # Search parent, CK+48, or ck subdirectories
        search_dirs = [p, p / "CK+48", p / "ck"]
        for emo_idx, emo_name in enumerate(EMOTION_LABELS):
            # Check folder name variations (happy/happiness, sad/sadness, angry/anger, etc.)
            variations = [emo_name]
            if emo_name == "happiness": variations.extend(["happy", "happiness"])
            if emo_name == "sadness": variations.extend(["sad", "sadness"])
            if emo_name == "surprise": variations.extend(["surprised", "surprise"])
            if emo_name == "anger": variations.extend(["angry", "anger", "angry_faces"])
            
            for s_dir in search_dirs:
                if not s_dir.exists():
                    continue
                for var in variations:
                    # Case insensitive search
                    emo_dir = s_dir / var
                    if not emo_dir.exists():
                        for child in s_dir.iterdir():
                            if child.is_dir() and child.name.lower() == var.lower():
                                emo_dir = child
                                break
                    
                    if emo_dir.exists():
                        files = list(emo_dir.glob("*.png")) + list(emo_dir.glob("*.jpg"))
                        for f in files:
                            combined_data.append((f, emo_idx))
                        if files:
                            print(f"  Loaded {len(files)} images for '{emo_name}' from CK+ subfolder '{emo_dir.name}'")

    # 2. Look for JAFFE dataset (search recursively via rglob)
    jaffe_path = dataset_paths.get("jaffe")
    if jaffe_path and os.path.exists(jaffe_path):
        print(f"[Dataset] Scanning JAFFE in: {jaffe_path}")
        jaffe_map = {
            "NE": 0, "HA": 1, "SU": 2, "SA": 3, "AN": 4, "DI": 5, "FE": 7,
        }
        p = Path(jaffe_path)
        files = list(p.rglob("*.tiff")) + list(p.rglob("*.jpg")) + list(p.rglob("*.png"))
        count = 0
        for f in files:
            name = f.name
            parts = name.split(".")
            if len(parts) >= 2:
                # Find matching emotion code in file parts, e.g. KA.AN1.39
                for part in parts:
                    if len(part) >= 2:
                        code = part[:2].upper()
                        if code in jaffe_map:
                            emo_idx = jaffe_map[code]
                            combined_data.append((f, emo_idx))
                            count += 1
                            break
        if count:
            print(f"  Loaded {count} images from JAFFE")

    # 3. Look for FER-2013 dataset (search train/test splits)
    fer_path = dataset_paths.get("fer2013")
    if fer_path and os.path.exists(fer_path):
        print(f"[Dataset] Scanning FER-2013 in: {fer_path}")
        p = Path(fer_path)
        for split in ["train", "test", "val", "validation"]:
            split_dir = p / split
            if split_dir.exists():
                for emo_idx, emo_name in enumerate(EMOTION_LABELS):
                    fer_folder_map = {
                        "neutral": 0, "happy": 1, "surprise": 2, "sad": 3, "angry": 4, "disgust": 5, "fear": 7
                    }
                    for folder_name, target_idx in fer_folder_map.items():
                        folder_dir = split_dir / folder_name
                        if not folder_dir.exists():
                            folder_dir = split_dir / str(list(fer_folder_map.keys()).index(folder_name))
                        
                        if folder_dir.exists():
                            imgs = list(folder_dir.glob("*.png")) + list(folder_dir.glob("*.jpg"))
                            for f in imgs:
                                combined_data.append((f, target_idx))
                                
    return combined_data


def train(args):
    if torch is None:
        print("[Error] PyTorch must be installed to run training.")
        return

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] Using hardware device: {device}")

    # Paths to look for datasets
    dataset_paths = {
        "ck": args.ck_dir,
        "jaffe": args.jaffe_dir,
        "fer2013": args.fer_dir
    }

    # Load data
    raw_data = scan_datasets(dataset_paths)
    use_mock = False
    
    if not raw_data:
        print("[Dataset] No raw dataset files found in specified directories.")
        print("          Proceeding with synthetic Mock Dataset for demonstration...")
        raw_data = generate_mock_data(num_samples=1200)
        use_mock = True
    else:
        print(f"[Dataset] Loaded total of {len(raw_data)} images from searched directories.")

    # Shuffle and Split (80% Train, 20% Val)
    np.random.seed(42)
    np.random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.8)
    train_raw = raw_data[:split_idx]
    val_raw = raw_data[split_idx:]

    train_dataset = EmotionTrainingDataset(train_raw, img_size=64, is_train=True)
    val_dataset = EmotionTrainingDataset(val_raw, img_size=64, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 1. Handle Class Imbalance using Loss Weights
    # Calculate inverse frequency weights
    labels_list = [item[1] for item in train_raw]
    class_counts = np.bincount(labels_list, minlength=8)
    print("[Class Distribution] Training counts per class:")
    for name, cnt in zip(EMOTION_LABELS, class_counts):
        print(f"  - {name:<10}: {cnt}")

    # Weighted Cross Entropy
    # weight_i = total_samples / (num_classes * count_i)
    total_samples = len(labels_list)
    class_weights = []
    for cnt in class_counts:
        w = total_samples / (8 * max(cnt, 1))
        class_weights.append(w)
    
    # Normalize weights so that minimum is 1.0
    min_w = min(class_weights)
    class_weights = [w / min_w for w in class_weights]
    print(f"[Weighted Loss] Computed class weights: {np.round(class_weights, 2)}")
    
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    # Model, Optimizer, Scheduler
    model = MiniXception(num_classes=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    # Training Loop
    print("\n[Train] Starting training epochs...")
    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        total_batches = len(train_loader)
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                current_loss = train_loss / total
                current_acc = (correct / total) * 100
                print(f"  -> Epoch {epoch:02d} | Batch {batch_idx + 1:04d}/{total_batches:04d} | Loss: {current_loss:.4f} Acc: {current_acc:.2f}%")
            
        epoch_loss = train_loss / total
        epoch_acc = correct / total
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total
        scheduler.step(val_epoch_loss)
        
        print(f"Epoch {epoch:02d}/{args.epochs:02d} | "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc*100:.2f}%")
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            # Save PyTorch checkpoint
            torch.save(model.state_dict(), "models/emotion_ferplus_best.pth")

    print(f"\n[Train] Completed! Best Val Accuracy: {best_acc*100:.2f}%")

    # 2. Export Model to ONNX Format
    print("[Export] Converting best model weights to ONNX...")
    model.load_state_dict(torch.load("models/emotion_ferplus_best.pth"))
    model.eval()
    
    # Dummy input representing (batch_size=1, channels=1, height=64, width=64)
    dummy_input = torch.randn(1, 1, 64, 64, device=device)
    
    onnx_path = "models/emotion_ferplus_custom.onnx" if not use_mock else "models/emotion_ferplus_mock.onnx"
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    print(f"[Export] Saved ONNX model to: {onnx_path}")
    print("          This file can be renamed to 'models/emotion_ferplus.onnx' to replace the default model.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train custom Facial Emotion Recognition model.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--fer_dir", type=str, default="data/fer2013", help="Directory path to FER-2013 dataset")
    parser.add_argument("--ck_dir", type=str, default="data/ckplus", help="Directory path to CK+ dataset")
    parser.add_argument("--jaffe_dir", type=str, default="data/jaffe", help="Directory path to JAFFE dataset")
    
    args = parser.parse_args()
    train(args)
