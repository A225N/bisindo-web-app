"""
Inti pipeline BISINDO untuk deploy.

Semua fungsi landmark di bawah disalin dari main_prominence.ipynb supaya fitur
yang masuk ke model identik dengan saat training (185 kolom per frame:
14 pose + 85 tangan kiri + 85 tangan kanan + 1 kolom jumlah tap).
Jangan ubah konstanta/fungsi di sini tanpa mengubah notebook training juga.
"""

import os
import sys

# Dipasang sebelum TensorFlow di-import: pakai CPU saja (sama seperti notebook)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from scipy.signal import find_peaks, savgol_filter


# ============================== KONSTANTA ==============================
# Urutan kelas = sorted() dari nama folder dataset (sama dengan notebook)
CLASSES = [
    "AKU", "APA", "BAGAIMANA", "BELAJAR", "BERANGKAT", "DATANG", "DIA", "DIMANA",
    "INGAT", "KAMU", "KAPAN", "KITA", "MAKAN", "MENGAPA", "MEREKA", "SIAPA",
]

SEQUENCE_LENGTH = 20

POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # hidung, bahu L/R, siku L/R, pergelangan L/R
POSE_DIM = len(POSE_IDX) * 2             # 14
HAND_DIM = 21 * 2 * 2 + 1                # 85
BASE_DIM = POSE_DIM + 2 * HAND_DIM       # 184
LANDMARK_DIM = BASE_DIM + 1              # 185 (+1 kolom jumlah tap)
LEFT_FLAG_POS = POSE_DIM + HAND_DIM - 1
RIGHT_FLAG_POS = BASE_DIM - 1

# HARUS sama dengan nilai yang dipakai saat membuat cache training
TAP_PROMINENCE = 0.3
TAP_SCALE = 4.0

SHOULDER_VIS_THRESHOLD = 0.3
MAX_PROCESS_WIDTH = 640


# ============================== UTILITAS ==============================
def resource_path(relative_path):
    """Path file yang bekerja saat dijalankan biasa maupun sebagai .exe (PyInstaller)."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative_path)


# ============================== LANDMARK (dari notebook) ==============================
def new_holistic():
    import mediapipe as mp  # import di sini agar modul ini bisa di-import tanpa mediapipe
    return mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def lm_xy(landmark_list, idx=None, aspect=1.0):
    pts = landmark_list.landmark
    if idx is not None:
        pts = [pts[i] for i in idx]
    return np.array([[p.x * aspect, p.y] for p in pts], dtype=np.float32)


def hand_vector(hand_landmarks, center, scale, aspect=1.0):
    if hand_landmarks is None:
        return np.zeros(HAND_DIM, dtype=np.float32)

    xy = lm_xy(hand_landmarks, aspect=aspect)
    body = (xy - center) / scale
    rel = xy - xy[0]
    extent = np.abs(rel).max()
    if extent > 1e-6:
        rel = rel / extent
    return np.concatenate([body.flatten(), rel.flatten(), [1.0]]).astype(np.float32)


def results_to_vector(results, last_state, aspect=1.0):
    if results.pose_landmarks is not None:
        lm = results.pose_landmarks.landmark
        if lm[11].visibility > SHOULDER_VIS_THRESHOLD and lm[12].visibility > SHOULDER_VIS_THRESHOLD:
            pts = lm_xy(results.pose_landmarks, POSE_IDX, aspect)
            l_shoulder, r_shoulder = pts[1], pts[2]
            scale = float(np.linalg.norm(l_shoulder - r_shoulder))
            if scale > 1e-3:
                center = (l_shoulder + r_shoulder) / 2.0
                pose_vec = ((pts - center) / scale).flatten().astype(np.float32)
                last_state = (center, scale, pose_vec)

    if last_state is None:
        return None, last_state

    center, scale, pose_vec = last_state
    left = hand_vector(results.left_hand_landmarks, center, scale, aspect)
    right = hand_vector(results.right_hand_landmarks, center, scale, aspect)

    return np.concatenate([pose_vec, left, right]).astype(np.float32), last_state


def select_indices(vectors, sequence=SEQUENCE_LENGTH):
    valid = [i for i, v in enumerate(vectors) if v is not None]
    if not valid:
        return None

    # tinggi tangan tertinggi per frame (y lebih kecil = lebih tinggi)
    # pose_vec: index 10,11 = wrist kiri (x,y), 12,13 = wrist kanan (x,y)
    h = np.array([min(vectors[i][11], vectors[i][13]) for i in valid])

    k = max(3, len(h) // 10)
    rest = max(np.median(h[:k]), np.median(h[-k:]))   # posisi tangan istirahat
    top = h.min()                                     # posisi tertinggi

    if rest - top > 0.3:                              # ada gerakan terangkat yang jelas
        thr = rest - 0.35 * (rest - top)
        raised = [valid[j] for j in range(len(valid)) if h[j] < thr]
        span = raised
        pad = 2                                       # sisakan sedikit awal/akhir gerakan
        start = max(valid[0], span[0] - pad)
        end = min(valid[-1], span[-1] + pad)
    else:
        # fallback: berdasarkan deteksi tangan
        active = [i for i in valid
                  if vectors[i][LEFT_FLAG_POS] == 1.0 or vectors[i][RIGHT_FLAG_POS] == 1.0]
        span = active if active else valid
        start, end = span[0], span[-1]

    return np.linspace(start, end, sequence).round().astype(int).tolist()


def forward_fill(vectors, indices):
    picked = []

    for i in indices:
        v = vectors[i]

        if v is None:
            j = i

            while j >= 0 and vectors[j] is None:
                j -= 1

            if j < 0:
                j = i

                while j < len(vectors) and vectors[j] is None:
                    j += 1

            v = vectors[j]
        picked.append(v)
    return np.stack(picked)


def count_taps(vectors, start, end):
    seg = forward_fill(vectors, list(range(start, end + 1)))
    y = seg[:, [11, 13]].mean(axis=1)          # y wrist kiri & kanan (sudah dinormalisasi bahu)
    if len(y) < 7:
        return 0
    y = savgol_filter(y, 7, 2)
    peaks, _ = find_peaks(-y, prominence=TAP_PROMINENCE, distance=3)
    return len(peaks)


def landmark_sequence(vectors, indices):
    seq = forward_fill(vectors, indices)
    taps = count_taps(vectors, indices[0], indices[-1]) / TAP_SCALE
    return np.hstack([seq, np.full((len(seq), 1), taps, np.float32)])


# ============================== DARI FRAME KAMERA KE FITUR ==============================
def process_frame(frame_bgr, holistic, last_state):
    height, width = frame_bgr.shape[:2]
    proc = frame_bgr

    if width > MAX_PROCESS_WIDTH:
        proc = cv2.resize(frame_bgr, (MAX_PROCESS_WIDTH, int(height * MAX_PROCESS_WIDTH / width)))

    results = holistic.process(cv2.cvtColor(proc, cv2.COLOR_BGR2RGB))
    vector, last_state = results_to_vector(results, last_state)
    return vector, last_state


def frames_to_landmarks(frames_bgr):
    """
    Daftar frame BGR hasil rekaman -> array (SEQUENCE_LENGTH, 185),
    atau None kalau pose tubuh tidak terdeteksi sama sekali.
    Frame diproses berurutan satu per satu, sama seperti read_and_detect di notebook.
    """
    if len(frames_bgr) == 0:
        return None

    vectors = []
    last_state = None

    with new_holistic() as holistic:
        for frame in frames_bgr:
            vector, last_state = process_frame(frame, holistic, last_state)
            vectors.append(vector)

    indices = select_indices(vectors, SEQUENCE_LENGTH)
    if indices is None:
        return None

    return landmark_sequence(vectors, indices)


# ============================== MODEL ==============================
def load_classifier(model_path):
    from tensorflow.keras.models import load_model  # import berat, dilakukan saat dibutuhkan

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"File model tidak ditemukan: {model_path}")

    model = load_model(model_path, compile=False)

    expected_dim = model.input_shape[-1]
    if expected_dim != LANDMARK_DIM:
        raise ValueError(
            f"Model mengharapkan {expected_dim} fitur per frame, kode ini menghasilkan {LANDMARK_DIM}. "
            f"Kemungkinan file .h5 berasal dari versi notebook yang berbeda."
        )

    num_outputs = model.output_shape[-1]
    if num_outputs != len(CLASSES):
        raise ValueError(f"Model punya {num_outputs} kelas, daftar CLASSES berisi {len(CLASSES)}.")

    return model


def predict_top3(model, landmark_seq):
    """Mengembalikan [(kata, probabilitas 0..1), ...] tiga teratas."""
    probs = model.predict(landmark_seq[None], verbose=0)[0]
    order = np.argsort(-probs)[:3]

    result = []
    for i in order:
        result.append((CLASSES[int(i)], float(probs[i])))
    return result