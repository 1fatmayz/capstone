import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from facenet_pytorch import InceptionResnetV1
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# SETTINGS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = r"C:\Users\ffatm\OneDrive\Desktop\capstone\Attendance-Face-Detection\clean_faces_4_white"
COLLECTION = "tryface"

MAX_PER_PERSON = 20

client = QdrantClient(host="localhost", port=6333)
client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
print(f"Recreated collection: {COLLECTION}")

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


def main():
    root = Path(DATA_DIR)

    if not root.exists():
        raise RuntimeError(f"DATA_DIR not found: {DATA_DIR}")

    person_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    print("People folders found:", len(person_dirs))

    point_id = 0
    enrolled_people = 0
    skipped_people = 0

    for person_dir in person_dirs:
        person_name = person_dir.name

        imgs = sorted([
            p for p in person_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if MAX_PER_PERSON is not None:
            imgs = imgs[:MAX_PER_PERSON]

        embs = []

        for img_path in imgs:
            emb = image_path_to_embedding(img_path)

            if emb is not None:
                embs.append(emb)

        if len(embs) == 0:
            skipped_people += 1
            print(f"Skipped {person_name}: no valid images")
            continue

        centroid = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        centroid = l2_normalize(centroid)

        # image_path is saved so Streamlit can show the most similar Qdrant face
        representative_image = str(imgs[0])

        point = PointStruct(
            id=point_id,
            vector=centroid.tolist(),
            payload={
                "student_id": str(point_id),
                "student_name": person_name,
                "n_imgs": len(embs),
                "image_path": representative_image,
                "source_image": representative_image,
            },
        )

        client.upsert(
            collection_name=COLLECTION,
            points=[point],
        )

        point_id += 1
        enrolled_people += 1

        print(f"Enrolled: {person_name} | images used: {len(embs)}")

    print("")
    print("=== DONE ENROLL ===")
    print("Enrolled people:", enrolled_people)
    print("Skipped people:", skipped_people)
    print("Collection:", COLLECTION)


if __name__ == "__main__":
    main()
