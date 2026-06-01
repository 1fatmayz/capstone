import os
import random
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1

# =========================
# SETTINGS (edit these)
# =========================
DATA_DIR = Path("clean_faces_4_white")  # your per-person folders
SAMPLES_PER_PERSON = 20                # take ~20 per person
SEED = 42

MODEL_WEIGHTS = "vggface2"             # "vggface2" is common & strong
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("embeddings_out")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Threshold search range (Euclidean distance on L2-normalized embeddings)
THRESH_MIN = 0.50
THRESH_MAX = 1.40
THRESH_STEP = 0.01
# =========================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def list_people(data_dir: Path):
    return sorted([p for p in data_dir.iterdir() if p.is_dir()])


def list_images(person_dir: Path):
    exts = {".jpg", ".jpeg", ".png"}
    imgs = [p for p in person_dir.iterdir() if p.suffix.lower() in exts]
    return sorted(imgs)


def load_img_160(path: Path) -> torch.Tensor:
    """
    Your cleaner outputs 160x160 with white background.
    We still enforce resize and normalization.
    """
    img = Image.open(path).convert("RGB").resize((160, 160))
    x = torch.from_numpy(np.asarray(img)).float()  # [H,W,C]
    x = x.permute(2, 0, 1) / 255.0                 # [C,H,W], 0..1
    # Facenet-pytorch typically expects [-1, 1]
    x = (x - 0.5) / 0.5
    return x


@torch.no_grad()
def embed_batch(model, batch: torch.Tensor) -> np.ndarray:
    batch = batch.to(DEVICE)
    emb = model(batch)  # [B,512]
    emb = emb.cpu().numpy().astype(np.float32)
    # L2 normalize (important for stable distance threshold)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return emb


def build_gallery_embeddings(model):
    """
    For each person: choose up to 20 images.
    We compute embeddings for each image and also a person "template" (mean).
    """
    people = list_people(DATA_DIR)
    labels = []
    paths = []
    all_embs = []

    print(f"Found {len(people)} person folders in {DATA_DIR.resolve()}")
    for person in people:
        imgs = list_images(person)
        if len(imgs) == 0:
            continue

        chosen = imgs if len(imgs) <= SAMPLES_PER_PERSON else random.sample(imgs, SAMPLES_PER_PERSON)

        batch = torch.stack([load_img_160(p) for p in chosen], dim=0)
        embs = embed_batch(model, batch)

        for pth, e in zip(chosen, embs):
            labels.append(person.name)
            paths.append(str(pth))
            all_embs.append(e)

        print(f"  {person.name}: used {len(chosen)} images")

    all_embs = np.stack(all_embs, axis=0) if len(all_embs) else np.zeros((0, 512), np.float32)
    labels = np.array(labels)
    paths = np.array(paths)

    # Person templates (mean embedding per person)
    templates = {}
    for person_name in sorted(set(labels.tolist())):
        idx = np.where(labels == person_name)[0]
        mean = all_embs[idx].mean(axis=0)
        mean /= (np.linalg.norm(mean) + 1e-12)
        templates[person_name] = mean.astype(np.float32)

    return all_embs, labels, paths, templates


def compute_pairs(all_embs: np.ndarray, labels: np.ndarray):
    """
    Build positive pairs (same person) and negative pairs (different person).
    For threshold tuning.
    """
    n = len(labels)
    if n < 2:
        return np.array([]), np.array([])

    # indices by label
    by_label = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)

    pos_dists = []
    neg_dists = []

    # positive: sample pairs within each label
    for lab, idxs in by_label.items():
        if len(idxs) < 2:
            continue
        # sample up to 100 pairs per class to avoid huge work
        pairs = []
        for _ in range(min(100, len(idxs) * 3)):
            a, b = random.sample(idxs, 2)
            pairs.append((a, b))
        for a, b in pairs:
            d = np.linalg.norm(all_embs[a] - all_embs[b])
            pos_dists.append(d)

    # negative: sample random cross-label pairs
    labs = list(by_label.keys())
    if len(labs) >= 2:
        for _ in range(min(5000, n * 5)):
            la, lb = random.sample(labs, 2)
            a = random.choice(by_label[la])
            b = random.choice(by_label[lb])
            d = np.linalg.norm(all_embs[a] - all_embs[b])
            neg_dists.append(d)

    return np.array(pos_dists, dtype=np.float32), np.array(neg_dists, dtype=np.float32)


def find_best_threshold(pos_dists: np.ndarray, neg_dists: np.ndarray):
    """
    Rule: accept "same person" if distance <= threshold.
    We maximize balanced accuracy between same/different.
    """
    if len(pos_dists) == 0 or len(neg_dists) == 0:
        return None

    best_t = None
    best_score = -1.0

    thresholds = np.arange(THRESH_MIN, THRESH_MAX + 1e-9, THRESH_STEP)
    for t in thresholds:
        tp = (pos_dists <= t).mean()  # true accept
        tn = (neg_dists > t).mean()   # true reject
        score = 0.5 * (tp + tn)       # balanced accuracy
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t, float(best_score)


def identify_from_templates(templates: dict, query_emb: np.ndarray, threshold: float):
    """
    1:N identification using template gallery.
    Returns: (pred_name or "unknown", best_distance)
    """
    best_name = None
    best_d = 1e9
    for name, tpl in templates.items():
        d = float(np.linalg.norm(query_emb - tpl))
        if d < best_d:
            best_d = d
            best_name = name

    if best_d <= threshold:
        return best_name, best_d
    return "unknown", best_d


def main():
    set_seed(SEED)

    # Load a strong pretrained face embedding model
    model = InceptionResnetV1(pretrained=MODEL_WEIGHTS).eval().to(DEVICE)
    print("Device:", DEVICE)
    print("Model:", f"InceptionResnetV1(pretrained='{MODEL_WEIGHTS}')")

    # 1) Build embeddings + templates (use ~20 images/person)
    all_embs, labels, paths, templates = build_gallery_embeddings(model)
    if len(labels) < 2:
        print("Not enough data. Need at least 2 images total.")
        return

    # Save raw embeddings
    np.save(OUT_DIR / "embeddings.npy", all_embs)
    np.save(OUT_DIR / "labels.npy", labels)
    np.save(OUT_DIR / "paths.npy", paths)

    with open(OUT_DIR / "templates.json", "w", encoding="utf-8") as f:
        # store as lists for json
        json.dump({k: v.tolist() for k, v in templates.items()}, f, indent=2)

    print(f"\nSaved embeddings to: {OUT_DIR.resolve()}")

    # 2) Tune threshold
    pos_dists, neg_dists = compute_pairs(all_embs, labels)
    res = find_best_threshold(pos_dists, neg_dists)
    if res is None:
        print("Could not compute threshold (need more same-person and different-person pairs).")
        return

    best_t, best_score = res
    print("\n=== THRESHOLD RESULT ===")
    print(f"Best threshold (Euclidean, L2-normalized): {best_t:.2f}")
    print(f"Balanced accuracy (same vs different):     {best_score:.3f}")

    with open(OUT_DIR / "threshold.json", "w", encoding="utf-8") as f:
        json.dump({"threshold": best_t, "balanced_accuracy": best_score}, f, indent=2)

    # 3) Quick demo: pick 5 random images and identify them
    print("\n=== QUICK IDENTIFY DEMO (templates) ===")
    for _ in range(5):
        i = random.randrange(len(paths))
        img_path = Path(paths[i])
        batch = torch.stack([load_img_160(img_path)], dim=0)
        q = embed_batch(model, batch)[0]
        pred, d = identify_from_templates(templates, q, best_t)
        print(f"GT={labels[i]:>15}  Pred={pred:>15}  dist={d:.3f}  file={img_path.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
