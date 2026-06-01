import sqlite3
import json
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import InceptionResnetV1


# SETTINGS
DB_PATH = "faces.db"
QUERY_IMAGE = "test.jpg"

THRESHOLD = 0.50   
TOP_K = 5          

# MODEL SETUP
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)

def image_to_embedding(img_path: str) -> np.ndarray:
    img = Image.open(img_path).convert("RGB").resize((160, 160))
    arr = np.asarray(img).copy()

    x = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = model(x).cpu().numpy()[0].astype(np.float32)

    return l2_normalize(emb)

# LOAD DATABASE (SQLite)
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

rows = c.execute("SELECT name, embedding FROM faces").fetchall()
conn.close()

if not rows:
    raise RuntimeError("No records found in faces table. Enroll faces first!")

print("Database loaded:")
print("Total records:", len(rows))

# QUERY IMAGE -> EMBEDDING
query_emb = image_to_embedding(QUERY_IMAGE)

# COSINE SIMILARITY + FILTER
matches = []  # (name, score)

for name, emb_str in rows:
    db_emb = np.array(json.loads(emb_str), dtype=np.float32)
    db_emb = l2_normalize(db_emb)

    score = float(np.dot(query_emb, db_emb))  

    if score > THRESHOLD:  
        matches.append((name, score))


matches.sort(key=lambda x: x[1], reverse=True)

print("\n==============================")
print(f"Query image: {QUERY_IMAGE}")
print(f"Threshold:   > {THRESHOLD}")
print("==============================\n")

if not matches:
    print("UNKNOWN (no match passed threshold)")
else:
    # show up to TOP_K passing matches
    shown = matches[:TOP_K]
    print(f" Matches above threshold (showing {len(shown)}):\n")
    for rank, (name, score) in enumerate(shown, 1):
        print(f"{rank}. {name:25s} cos={score:.3f}")

    # Best match decision
    best_name, best_score = matches[0]
    print("\nBest match:", best_name)
    print("Best cosine:", best_score)
    print(" KNOWN (passed threshold)")
