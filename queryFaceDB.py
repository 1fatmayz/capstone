import os
import cv2
import numpy as np
import torch
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# SETTINGS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

QUERY_DIR = r"C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection\clean_faces_4_white"
COLLECTION = "tryface"

THRESHOLD = 0.72     
TOP_K = 5
GAP = 0.05

MAX_QUERIES = 50      


client = QdrantClient(host="localhost", port=6333)
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def image_path_to_embedding(img_path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (160, 160), interpolation=cv2.INTER_AREA)

    arr = img_rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1)
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(x).detach().cpu().numpy()[0].astype(np.float32)

    return l2_normalize(emb)

def qdrant_search(emb: np.ndarray, top_k: int):
    res = client.query_points(
        collection_name=COLLECTION,
        query=emb.tolist(),
        limit=top_k,
        with_payload=True
    )
    return res.points

def main():
    if not os.path.isdir(QUERY_DIR):
        raise RuntimeError(f"Query folder not found: {QUERY_DIR}")

    print("====================================")
    print("Query folder:", QUERY_DIR)
    print("Collection  :", COLLECTION)
    print("Threshold   :", f"> {THRESHOLD}")
    print("TOP_K       :", TOP_K)
    print("GAP         :", GAP)
    print("MAX_QUERIES :", MAX_QUERIES)
    print("Device      :", DEVICE)
    print("====================================\n")

    image_paths = sorted([p for p in Path(QUERY_DIR).rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    print("Total images found:", len(image_paths))

    tested = 0
    for img_path in image_paths:
        if MAX_QUERIES is not None and tested >= MAX_QUERIES:
            break

        emb = image_path_to_embedding(img_path)
        if emb is None:
            continue

        results = qdrant_search(emb, TOP_K)
        tested += 1

        print("\n==============================")
        print(f"Query image: {img_path}")
        print("==============================")

        if not results                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    :
            print(" UNKNOWN (no candidates)")
            continue

        scored = []
        for r in results:
            score = float(r.score)
            name = (r.payload or {}).get("student_name", "UNKNOWN_NAME_IN_PAYLOAD")
            scored.append((name, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        top1_name, top1 = scored[0]
        top2 = scored[1][1] if len(scored) > 1 else -1.0
        gap = top1 - top2

        print("Top candidates:")
        for i, (n, s) in enumerate(scored, 1):
            print(f"{i}. {n:25s} cos={s:.3f}")

        if top1 > THRESHOLD and gap >= GAP:
            print(f"\n KNOWN: {top1_name}")
            print(f"Best cosine: {top1:.3f}  | gap: {gap:.3f}")
        else:
            print(f"\n UNKNOWN (too low / ambiguous)")
            print(f"top1={top1:.3f}, top2={top2:.3f}, gap={gap:.3f}")

if __name__ == "__main__":
    main()
