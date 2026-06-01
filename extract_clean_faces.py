import gc
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from facenet_pytorch import MTCNN

INPUT_DIR = Path(r"my_dataset")
OUTPUT_DIR = Path(r"clean_faces_4_white")

IMAGE_SIZE = 160
EXTRA_BOX_SCALE = 0.60

MAX_SIDE = 650
MAX_CROP_SIDE = 520

PROB_THRESHOLD = 0.75

# GrabCut
GRABCUT_ITERS = 3
FEATHER = 3

MIN_MASK_AREA_RATIO = 0.06

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

mtcnn = MTCNN(
    image_size=IMAGE_SIZE,
    margin=0,
    select_largest=True,
    post_process=True,
    keep_all=False,
    device="cpu"
)

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def safe_resize_max_side(img_bgr, max_side):
    h, w = img_bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img_bgr, 1.0
    s = max_side / m
    new_w = max(1, int(w * s))
    new_h = max(1, int(h * s))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, s

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

    if nx2 <= nx1: nx2 = clamp(nx1 + 1, 0, w - 1)
    if ny2 <= ny1: ny2 = clamp(ny1 + 1, 0, h - 1)
    return nx1, ny1, nx2, ny2

def align_face_bgr(img_bgr, landmarks):
    le = landmarks[0]
    re = landmarks[1]

    desired_left = (0.32 * IMAGE_SIZE, 0.35 * IMAGE_SIZE)
    desired_right = (0.68 * IMAGE_SIZE, 0.35 * IMAGE_SIZE)

    dx = re[0] - le[0]
    dy = re[1] - le[1]
    dist = float(np.sqrt(dx * dx + dy * dy) + 1e-6)

    desired_dist = float(desired_right[0] - desired_left[0])
    scale = desired_dist / dist
    angle = float(np.degrees(np.arctan2(dy, dx)))

    eyes_center = ((le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale)

    desired_center = ((desired_left[0] + desired_right[0]) / 2.0, desired_left[1])
    M[0, 2] += (desired_center[0] - eyes_center[0])
    M[1, 2] += (desired_center[1] - eyes_center[1])

    aligned = cv2.warpAffine(
        img_bgr, M, (IMAGE_SIZE, IMAGE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderValue=(255, 255, 255)
    )
    return aligned

def ellipse_mask(h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w//2, h//2), (int(w*0.44), int(h*0.58)), 0, 0, 360, 255, -1)
    return m

def grabcut_mask(img_bgr):
    h, w = img_bgr.shape[:2]

    init = ellipse_mask(h, w)

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[init == 255] = cv2.GC_PR_FGD

    bgModel = np.zeros((1, 65), np.float64)
    fgModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, gc_mask, None, bgModel, fgModel, GRABCUT_ITERS, cv2.GC_INIT_WITH_MASK)

    mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    if float((mask > 0).mean()) < MIN_MASK_AREA_RATIO:
        mask = init

    return mask

def feather_mask(mask, feather=3):
    if feather <= 0:
        return mask
    k = feather * 2 + 1
    return cv2.GaussianBlur(mask, (k, k), 0)

def composite_on_white(img_bgr, mask_soft):
    m = (mask_soft.astype(np.float32) / 255.0)[..., None]
    white = np.full_like(img_bgr, 255, dtype=np.uint8)
    return (img_bgr.astype(np.float32) * m + white.astype(np.float32) * (1 - m)).astype(np.uint8)

def center_crop_square(img_bgr):
    h, w = img_bgr.shape[:2]
    side = min(h, w)
    y1 = (h - side) // 2
    x1 = (w - side) // 2
    return img_bgr[y1:y1+side, x1:x1+side]

def make_white_output(best_bgr):
    """
    Guarantee white background:
    - resize to IMAGE_SIZE
    - grabcut mask
    - feather
    - composite on white
    """
    img = cv2.resize(best_bgr, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    try:
        mask = grabcut_mask(img)
        mask_soft = feather_mask(mask, FEATHER)
        out = composite_on_white(img, mask_soft)
    except Exception:
        
        m = feather_mask(ellipse_mask(IMAGE_SIZE, IMAGE_SIZE), FEATHER)
        out = composite_on_white(img, m)
    return out

def process_one_image(img_path: Path, out_path: Path) -> bool:
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        return False  

    img_bgr, _ = safe_resize_max_side(img_bgr, MAX_SIDE)
    h, w = img_bgr.shape[:2]


    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    boxes, probs, landmarks = mtcnn.detect(img_pil, landmarks=True)

    if boxes is None or probs is None or len(boxes) == 0:
        
        crop = center_crop_square(img_bgr)
        out = make_white_output(crop)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
        return True

    best_i = int(np.argmax(probs))
    if float(probs[best_i]) < PROB_THRESHOLD:
        
        crop = center_crop_square(img_bgr)
        out = make_white_output(crop)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
        return True

    box = boxes[best_i]
    lm = landmarks[best_i]

    
    x1, y1, x2, y2 = expand_box(box, w, h, EXTRA_BOX_SCALE)
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        crop = center_crop_square(img_bgr)
        out = make_white_output(crop)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), out)
        return True

    
    lm2 = lm.astype(np.float32).copy()
    lm2[:, 0] -= x1
    lm2[:, 1] -= y1

    
    rh, rw = roi.shape[:2]
    if max(rh, rw) > MAX_CROP_SIDE:
        roi, s = safe_resize_max_side(roi, MAX_CROP_SIDE)
        lm2 *= s

    roi = roi.copy()

    
    temp = 320
    roi_temp = cv2.resize(roi, (temp, temp), interpolation=cv2.INTER_AREA)

    sx = temp / roi.shape[1]
    sy = temp / roi.shape[0]
    lm_temp = lm2.copy()
    lm_temp[:, 0] *= sx
    lm_temp[:, 1] *= sy

    aligned = align_face_bgr(roi_temp, lm_temp)


    out = make_white_output(aligned)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)

    del img_bgr, img_rgb, img_pil, roi, roi_temp, aligned, out
    gc.collect()
    return True

def main():
    print("INPUT_DIR:", INPUT_DIR.resolve())
    print("INPUT_DIR exists:", INPUT_DIR.exists())
    if not INPUT_DIR.exists():
        print("ERROR: Input folder not found:", INPUT_DIR.resolve())
        return

    total = saved = unreadable = 0
    person_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir()]
    print(f"\nFound {len(person_folders)} person folders in: {INPUT_DIR}")

    for idx, person_folder in enumerate(person_folders, 1):
        out_person = OUTPUT_DIR / person_folder.name
        out_person.mkdir(parents=True, exist_ok=True)

        imgs = [p for p in person_folder.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]

        for img_path in imgs:
            total += 1
            out_path = out_person / img_path.name
            ok = process_one_image(img_path, out_path)
            if ok:
                saved += 1
            else:
                unreadable += 1

        if idx % 5 == 0:
            print(f"[{idx}/{len(person_folders)}] saved={saved} unreadable={unreadable}")

    print("\n=== DONE ===")
    print("Total images read:", total)
    print("Saved (always white):", saved)
    print("Unreadable only:", unreadable)
    print("Output folder:", OUTPUT_DIR.resolve())

if __name__ == "__main__":
    main()
