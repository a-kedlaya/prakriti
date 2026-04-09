import os
import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

# ── Load prakriti model ───────────────────────────────────────────────────────
prakriti_model = load_model("prakriti_model.h5", compile=False)
IMG_SIZE = 224
classes  = ['Kapha', 'Pitta', 'Pitta-Kapha', 'Vata', 'Vata-Kapha', 'Vata-Pitta']

# ── Load face detector ────────────────────────────────────────────────────────
face_net = cv2.dnn.readNetFromCaffe(
    "age_model/deploy.prototxt",
    "age_model/res10_300x300_ssd_iter_140000.caffemodel"
)

# ── Age (OpenCV bucket model) ─────────────────────────────────────────────────
age_net = cv2.dnn.readNet(
    "age_model/age_net.caffemodel",
    "age_model/age_deploy.prototxt"
)
MODEL_MEAN    = (78.4263377603, 87.7689143744, 114.895847746)
AGE_MIDPOINTS = [1, 5, 10, 17, 28, 40, 50, 70]

# ── Gender — GenderNet Caffe (no DeepFace dependency) ────────────────────────
GENDER_MODEL = "age_model/gender_net.caffemodel"
GENDER_PROTO = "age_model/gender_deploy.prototxt"
gender_net   = None
if os.path.exists(GENDER_MODEL) and os.path.exists(GENDER_PROTO):
    gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)
    print("Gender model loaded.")
else:
    print("Gender model not found. Run: python download_gender_model.py")

GENDER_LIST = ['Male', 'Female']

# ── Age regression model ──────────────────────────────────────────────────────
AGE_REGRESSION_PATH = "age_model/age_regression.h5"
age_regression      = None
AGE_REG_SIZE        = 64
if os.path.exists(AGE_REGRESSION_PATH):
    age_regression = load_model(AGE_REGRESSION_PATH, compile=False)
    age_regression.predict(np.zeros((1, AGE_REG_SIZE, AGE_REG_SIZE, 3)), verbose=0)
    print("Age regression model loaded.")
else:
    print("Using bucket age model. Run python train_age_model.py for better accuracy.")

# ── Warm up prakriti model ────────────────────────────────────────────────────
prakriti_model.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3)), verbose=0)
print("Prakriti model warmed up.")

# ── Camera ────────────────────────────────────────────────────────────────────
cap = None
for idx in range(3):
    _c = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if _c.isOpened():
        ok, _ = _c.read()
        if ok:
            cap = _c
            print(f"Camera opened at index {idx}")
            break
        _c.release()

if cap is None:
    print("ERROR: No camera found.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

# ── Smoothing buffers (deque = auto-drop old values) ─────────────────────────
SMOOTH_N     = 10   # frames to average over — higher = more stable
INFER_EVERY  = 6    # run heavy inference every N frames

age_buf      = deque(maxlen=SMOOTH_N)   # smoothed age values
gender_buf   = deque(maxlen=20)   # larger buffer = more stable gender
prakriti_buf = deque(maxlen=SMOOTH_N)   # softmax score arrays

frame_count  = 0
last_display = None  # cached display result

# ── Inference helpers ─────────────────────────────────────────────────────────
def run_age(face_bgr):
    if age_regression is not None:
        face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (AGE_REG_SIZE, AGE_REG_SIZE)).astype(np.float32) / 255.0
        age  = float(age_regression.predict(np.expand_dims(face, 0), verbose=0)[0][0])
        return np.clip(age, 1, 100)
    blob = cv2.dnn.blobFromImage(face_bgr, 1.0, (227, 227), MODEL_MEAN)
    age_net.setInput(blob)
    probs = age_net.forward()[0]
    return float(np.dot(probs, AGE_MIDPOINTS))

def run_gender(face_bgr):
    """
    Returns ('Male'|'Female', confidence%) using GenderNet Caffe.
    Runs inference on 3 crops (center + slight zoom) and averages
    probabilities — reduces single-frame noise significantly.
    """
    if gender_net is None:
        return None, 0.0

    h, w = face_bgr.shape[:2]
    crops = [
        face_bgr,                                                        # full face
        face_bgr[int(h*0.05):int(h*0.95), int(w*0.05):int(w*0.95)],   # slight crop
        face_bgr[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)],   # tighter crop
    ]
    all_probs = []
    for crop in crops:
        if crop.size == 0:
            continue
        blob = cv2.dnn.blobFromImage(crop, 1.0, (227, 227), MODEL_MEAN)
        gender_net.setInput(blob)
        all_probs.append(gender_net.forward()[0])

    if not all_probs:
        return None, 0.0

    avg_probs = np.mean(all_probs, axis=0)
    idx  = int(np.argmax(avg_probs))
    conf = float(avg_probs[idx]) * 100
    return GENDER_LIST[idx], conf

def run_prakriti(face_bgr):
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    return prakriti_model.predict(np.expand_dims(face, 0), verbose=0)[0]

def draw_led(frame, center, color, on):
    cv2.circle(frame, center, 18, color, -1 if on else 2)

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        cv2.waitKey(30)
        continue

    frame_count += 1
    h, w   = frame.shape[:2]
    display = frame.copy()

    # Face detection every frame (fast)
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    face_net.setInput(blob)
    dets = face_net.forward()

    vata_on = pitta_on = kapha_on = False
    face_found = False

    for i in range(dets.shape[2]):
        conf = dets[0, 0, i, 2]
        if conf < 0.6:
            continue

        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        x1, y1, x2, y2 = np.clip(box.astype(int), 0, [w, h, w, h])
        face = frame[y1:y2, x1:x2]
        if face.size == 0 or face.shape[0] < 40 or face.shape[1] < 40:
            continue

        face_found = True

        # ── Run inference every INFER_EVERY frames ────────────────────────────
        if frame_count % INFER_EVERY == 0:
            # Age — push raw float into buffer
            age_raw = run_age(face)
            age_buf.append(age_raw)

            # Gender — only buffer if confidence > 85% to prevent flipping
            gender_label, gender_conf = run_gender(face)
            if gender_label is not None and gender_conf > 85.0:
                gender_buf.append((gender_label, gender_conf))

            # Prakriti — push softmax array into buffer
            scores = run_prakriti(face)
            prakriti_buf.append(scores)

        # ── Compute stable values from buffers ────────────────────────────────
        # Age: median of buffer (robust to outliers) then round to nearest 2
        if age_buf:
            stable_age = int(round(np.median(age_buf) / 2) * 2)
        else:
            stable_age = 0

        # Gender: weighted majority vote — higher confidence = more weight
        if gender_buf:
            male_score   = sum(c for l, c in gender_buf if l == 'Male')
            female_score = sum(c for l, c in gender_buf if l == 'Female')
            stable_gender      = 'Male' if male_score >= female_score else 'Female'
            stable_gender_conf = max(male_score, female_score) / len(gender_buf)
        else:
            stable_gender      = None
            stable_gender_conf = 0.0

        # Prakriti: mean of softmax buffer
        if prakriti_buf:
            smoothed     = np.mean(prakriti_buf, axis=0)
            prakriti_idx = int(np.argmax(smoothed))
            prakriti     = classes[prakriti_idx]
            confidence   = float(smoothed[prakriti_idx]) * 100
        else:
            continue

        # LED logic
        if "Vata"  in prakriti: vata_on  = True
        if "Pitta" in prakriti: pitta_on = True
        if "Kapha" in prakriti: kapha_on = True

        # ── Draw ──────────────────────────────────────────────────────────────
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Line 1: Prakriti + confidence
        cv2.putText(display, f"{prakriti}  ({confidence:.0f}%)",
                    (x1, max(y1 - 30, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Line 2: Age + Gender
        age_gender = f"Age: ~{stable_age}"
        if stable_gender:
            age_gender += f"   {stable_gender}"
        cv2.putText(display, age_gender,
                    (x1, max(y1 - 8, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2)

        # Top-3 prakriti scores (small, below box)
        top3 = np.argsort(smoothed)[::-1][:3]
        for rank, cidx in enumerate(top3):
            cv2.putText(display,
                        f"{classes[cidx]}: {smoothed[cidx]*100:.1f}%",
                        (x1, min(y2 + 18 + rank * 16, h - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 0), 1)

    # ── LED panel ─────────────────────────────────────────────────────────────
    draw_led(display, (38,  38), (60,  60,  255), vata_on)
    draw_led(display, (95,  38), (0,   230, 230), pitta_on)
    draw_led(display, (152, 38), (255, 80,  80),  kapha_on)
    for txt, cx in [("Vata", 20), ("Pitta", 77), ("Kapha", 134)]:
        cv2.putText(display, txt, (cx, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)

    cv2.imshow("Prakriti Bot", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
