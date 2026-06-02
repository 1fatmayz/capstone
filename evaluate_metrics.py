import random
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = Path(r"C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection\clean_faces_4_white")

THRESHOLD = 0.72
GAP = 0.05
TOP_K = 5
IMAGE_SIZE = 160
TEST_RATIO = 0.30
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def l2_normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)


def image_to_embedding(img_path):
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)

    arr = img_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(x).detach().cpu().numpy()[0].astype(np.float32)

    return l2_normalize(emb)


def load_dataset():
    people = sorted([p for p in DATA_DIR.iterdir() if p.is_dir()])
    data = {}

    for person in people:
        images = sorted([
            p for p in person.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if len(images) >= 2:
            data[person.name] = images

    return data


def split_train_test(data):
    train = {}
    test = {}

    for name, images in data.items():
        images = images.copy()
        random.shuffle(images)

        test_count = max(1, int(len(images) * TEST_RATIO))

        test[name] = images[:test_count]
        train[name] = images[test_count:]

        if len(train[name]) == 0:
            train[name] = test[name][:1]

    return train, test


def build_gallery(train):
    gallery = {}

    print("Building gallery embeddings...")

    for name, images in train.items():
        embs = []

        for img_path in images:
            emb = image_to_embedding(img_path)
            if emb is not None:
                embs.append(emb)

        if len(embs) > 0:
            centroid = np.mean(np.stack(embs), axis=0).astype(np.float32)
            gallery[name] = l2_normalize(centroid)

        print(f"{name}: {len(embs)} train images")

    return gallery


def recognize(query_emb, gallery):
    scores = []

    for name, gallery_emb in gallery.items():
        score = float(np.dot(query_emb, gallery_emb))
        scores.append((name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    scores = scores[:TOP_K]

    top1_name, top1 = scores[0]
    top2 = scores[1][1] if len(scores) > 1 else -1.0
    gap = top1 - top2

    if top1 > THRESHOLD and gap >= GAP:
        return top1_name, top1, gap

    return "UNKNOWN", top1, gap


def evaluate():
    data = load_dataset()

    if len(data) < 2:
        raise RuntimeError("Need at least 2 people folders with images.")

    train, test = split_train_test(data)
    gallery = build_gallery(train)

    y_true = []
    y_pred = []

    total = 0
    correct_identity = 0
    unknown_count = 0

    print("\nTesting recognition...\n")

    for true_name, images in test.items():
        for img_path in images:
            emb = image_to_embedding(img_path)
            if emb is None:
                continue

            pred_name, score, gap = recognize(emb, gallery)

            y_true.append(true_name)
            y_pred.append(pred_name)

            total += 1

            if pred_name == true_name:
                correct_identity += 1

            if pred_name == "UNKNOWN":
                unknown_count += 1

            print(f"Image: {img_path.name}")
            print(f"True: {true_name}")
            print(f"Predicted: {pred_name}")
            print(f"Score: {score:.3f} | Gap: {gap:.3f}")
            print("-" * 40)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n==============================")
    print("FINAL EVALUATION RESULTS")
    print("==============================")
    print(f"Total test images: {total}")
    print(f"Correct predictions: {correct_identity}")
    print(f"Unknown predictions: {unknown_count}")
    print("")
    print(f"Accuracy:  {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall:    {recall * 100:.2f}%")
    print(f"F1-Score:  {f1 * 100:.2f}%")

    print("\nConfusion Matrix:")
    labels = sorted(list(set(y_true + y_pred)))
    print(labels)
    print(confusion_matrix(y_true, y_pred, labels=labels))


if __name__ == "__main__":
    evaluate()
