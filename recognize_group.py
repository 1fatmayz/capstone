import os
import cv2
import cv2
import uuid
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_PATH = "captured.jpg"

COLLECTION = "tryface"

TOP_K = 5
THRESHOLD = 0.72
GAP = 0.05

MIN_FACE_SIZE = 60
PROB_THRESHOLD = 0.70

IMAGE_SIZE = 160
EXTRA_BOX_SCALE = 0.60

SAVE_CROPS = True
CROPS_DIR = Path("extracted_faces_from_group")

SAVE_TO_QDRANT = True

ONLY_SAVE_KNOWN = True

FORCE_WHITE_BG = True
FEATHER = 3
# 



client = QdrantClient(host="localhost", port=6333)

mtcnn_all = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    keep_all=True,
    post_process=True,
    min_face_size=MIN_FACE_SIZE,
    device=DEVICE,
)

mtcnn_one = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    keep_all=False,
    post_process=True,
    min_face_size=MIN_FACE_SIZE,
    device=DEVICE,
)

model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)



def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def expand_box(box, w, h, scale=0.60):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    new_w = bw * (1.0 + scale)
    new_h = bh * (1.0 + scale)

    cy = cy - 0.08 * bh

    nx1 = clamp(int(cx - new_w / 2.0), 0, w - 1)
    ny1 = clamp(int(cy - new_h / 2.0), 0, h - 1)
    nx2 = clamp(int(cx + new_w / 2.0), 0, w - 1)
    ny2 = clamp(int(cy + new_h / 2.0), 0, h - 1)

    if nx2 <= nx1:
        nx2 = clamp(nx1 + 1, 0, w - 1)
    if ny2 <= ny1:
        ny2 = clamp(ny1 + 1, 0, h - 1)

    return nx1, ny1, nx2, ny2


def ellipse_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(
        m,
        (w // 2, h // 2),
        (int(w * 0.44), int(h * 0.58)),
        0,
        0,
        360,
        255,
        -1,
    )
    return m


def feather_mask(mask, feather=3):
    if feather <= 0:
        return mask
    k = feather * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)


def composite_on_white(img_bgr, mask_soft):
    m = (mask_soft.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(img_bgr, 255, dtype=np.uint8)
    return (img_bgr.astype(np.float32) * m + white.astype(np.float32) * (1 - m)).astype(np.uint8)


def force_white_background(face_bgr_160):
    m = feather_mask(ellipse_mask(IMAGE_SIZE, IMAGE_SIZE), FEATHER)
    return composite_on_white(face_bgr_160, m)


def face_tensor_to_bgr160(face_tensor: torch.Tensor) -> np.ndarray:
    face_rgb = face_tensor.permute(1, 2, 0).detach().cpu().numpy()
    face_rgb = (face_rgb * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)


def face_bgr_to_embedding(face_bgr_160: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(face_bgr_160, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)
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
        with_payload=True,
    )
    return res.points


def decide_name(points):
    if not points:
        return "UNKNOWN", -1.0, 0.0, []

    scored = []
    for p in points:
        score = float(p.score)
        payload = p.payload or {}
        found_name = payload.get("student_name", "UNKNOWN_NAME_IN_PAYLOAD")
        scored.append((found_name, score))

    scored.sort(key=lambda x: x[1], reverse=True)

    top1_name, top1 = scored[0]
    top2 = scored[1][1] if len(scored) > 1 else -1.0
    gap = top1 - top2

    if top1 > THRESHOLD and gap >= GAP:
        return top1_name, top1, gap, scored

    return "UNKNOWN", top1, gap, scored


def make_safe_filename(name: str) -> str:
    safe = name.strip()
    safe = safe.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace(":", "_")
    safe = safe.replace("*", "_")
    safe = safe.replace("?", "_")
    safe = safe.replace('"', "_")
    safe = safe.replace("<", "_")
    safe = safe.replace(">", "_")
    safe = safe.replace("|", "_")
    return safe


def save_face_to_qdrant(embedding: np.ndarray, student_name: str, image_path: str, face_number: int):
    point = PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            "student_name": student_name,
            "image_path": image_path,
            "source_image": IMAGE_PATH,
            "face_number": face_number,
        }
    )

    client.upsert(
        collection_name=COLLECTION,
        points=[point]
    )


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open Logitech camera")
    exit()

print("Press SPACE to capture image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("No frame received")
        continue

    cv2.imshow("Logitech Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    
    if key == 32:
        cv2.imwrite("captured.jpg", frame)
        print("Image captured successfully!")
        break

    
    elif key == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

# MAIN
def main():
    if not os.path.isfile(IMAGE_PATH):
        raise RuntimeError(f"Image not found: {IMAGE_PATH}")

    img_bgr = cv2.imread(IMAGE_PATH, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {IMAGE_PATH}")

    h, w = img_bgr.shape[:2]

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    boxes, probs, _ = mtcnn_all.detect(img_pil, landmarks=True)

    if boxes is None or probs is None or len(boxes) == 0:
        print(" No faces detected in group image.")
        return

    valid_idxs = [i for i in range(len(boxes)) if float(probs[i]) >= PROB_THRESHOLD]
    if not valid_idxs:
        print(" Faces detected but all below PROB_THRESHOLD.")
        return

    if SAVE_CROPS:
        CROPS_DIR.mkdir(parents=True, exist_ok=True)

    print("====================================")
    print("Image       :", IMAGE_PATH)
    print("Collection  :", COLLECTION)
    print("Faces found :", len(boxes))
    print("Valid faces :", len(valid_idxs))
    print("Threshold   :", THRESHOLD)
    print("GAP         :", GAP)
    print("Device      :", DEVICE)
    print("====================================")

    per_face_predictions = []

    for face_num, idx in enumerate(valid_idxs, 1):
        x1, y1, x2, y2 = expand_box(boxes[idx], w, h, EXTRA_BOX_SCALE)
        crop_bgr = img_bgr[y1:y2, x1:x2]

        if crop_bgr.size == 0:
            per_face_predictions.append("UNKNOWN")
            print(f"\n--- Face #{face_num} ---")
            print("Predicted: UNKNOWN (empty crop)")
            continue

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        aligned_tensor = mtcnn_one(crop_rgb)

        if aligned_tensor is None:
            face_bgr_160 = cv2.resize(crop_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        else:
            face_bgr_160 = face_tensor_to_bgr160(aligned_tensor)

        if FORCE_WHITE_BG:
            face_bgr_160 = force_white_background(face_bgr_160)

        emb = face_bgr_to_embedding(face_bgr_160)
        points = qdrant_search(emb, TOP_K)

        name, top1, gap, scored = decide_name(points)
        per_face_predictions.append(name)

        print(f"\n--- Face #{face_num} ---")
        print("Predicted:", name)

        if scored:
            print("Top candidates:")
            for r, (n, s) in enumerate(scored, 1):
                print(f"{r}. {n:25s} cos={s:.3f}")
            print(f"top1={top1:.3f}, gap={gap:.3f}")

        image_path = None

        if SAVE_CROPS:
            safe_name = make_safe_filename(name)
            out_face = CROPS_DIR / f"{safe_name}_{face_num:02d}.jpg"
            cv2.imwrite(str(out_face), face_bgr_160)
            image_path = str(out_face.resolve())
            print("Saved image locally as:", out_face.name)

        should_save = SAVE_TO_QDRANT and image_path is not None

        if ONLY_SAVE_KNOWN:
            should_save = should_save and (name != "UNKNOWN")

        if should_save:
            save_face_to_qdrant(
                embedding=emb,
                student_name=name,
                image_path=image_path,
                face_number=face_num
            )
            print("Saved to Qdrant with person name:", name)
            print("Saved image path:", image_path)
        else:
            print("Not saved to Qdrant.")

    unique_known = sorted({n for n in per_face_predictions if n != "UNKNOWN"})

    print("\n==============================")
    print("FINAL OUTPUT (unique names):")
    if unique_known:
        for n in unique_known:
            print("-", n)
    else:
        print("No confident matches (all UNKNOWN).")

    if SAVE_CROPS:
        print("\nSaved extracted aligned faces locally to:")
        print(CROPS_DIR.resolve())


if __name__ == "__main__":
    main()
