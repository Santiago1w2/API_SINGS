import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pickle

from keras.models import load_model
from helpers import (
    get_word_ids,
    mediapipe_detection,
    there_hand,
    extract_keypoints
)
from constants import *

# ─────────────────────────────────────────────
# IMPORTANTE: mismos parámetros exactos
# ─────────────────────────────────────────────

THRESHOLD = 0.80
CONFIDENCE_MARGIN = 0.15
MIN_FRAMES = 15
MAX_FRAMES = 70

STATIC_THRESHOLD = 0.10
STATIC_FRAMES = 5
RF_CONFIRM_FRAMES = 3
STILLNESS_FRAMES = 10
RF_ACTIVE_MAX_LEN = 15
COOLDOWN_FRAMES = 2


# ─────────────────────────────────────────────
# RF
# ─────────────────────────────────────────────

def load_rf():
    rf1 = rf2 = None
    try:
        with open("model_1hand.p", "rb") as f:
            rf1 = pickle.load(f)
    except:
        pass

    try:
        with open("model_2hands.p", "rb") as f:
            rf2 = pickle.load(f)
    except:
        pass

    return rf1, rf2


def count_active_hands(kp):
    return int(not np.allclose(kp[:63], 0)) + int(not np.allclose(kp[63:], 0))


def is_still(a, b):
    if b is None:
        return False
    return np.mean(np.abs(np.array(a) - np.array(b))) < STATIC_THRESHOLD


# ─────────────────────────────────────────────
# LSTM helper
# ─────────────────────────────────────────────

def interpolate(kp, target=25):
    if len(kp) == target:
        return kp
    idx = np.linspace(0, len(kp) - 1, target)
    out = []
    for i in idx:
        l, u = int(np.floor(i)), int(np.ceil(i))
        if l == u:
            out.append(kp[l])
        else:
            w = i - l
            out.append((1-w)*np.array(kp[l]) + w*np.array(kp[u]))
    return out


def normalize(kp):
    if len(kp) > 25:
        step = len(kp)/25
        idx = np.arange(0, len(kp), step).astype(int)[:25]
        return [kp[i] for i in idx]
    return interpolate(kp)


# ─────────────────────────────────────────────
# ENGINE PRINCIPAL (MISMO FLUJO QUE evaluate_model)
# ─────────────────────────────────────────────

def run_pipeline(video_path):
    word_ids = get_word_ids(WORDS_JSON_PATH)

    lstm = load_model(MODEL_PATH)
    predict_fn = tf.function(lstm, reduce_retracing=True)

    rf1, rf2 = load_rf()
    rf_available = rf1 is not None or rf2 is not None

    kp_seq = []
    sentence = []

    count_frame = 0
    still_count = 0
    static_count = 0
    cooldown = 0
    prev_kp = None
    lost_hand_frames = 0
    rf_last_letter = None
    rf_confirm = 0

    cap = cv2.VideoCapture(video_path)

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = mediapipe_detection(frame, hands)
            hand = there_hand(results)

            # ── timeout
            if len(kp_seq) > MAX_FRAMES:
                kp_seq = []
                count_frame = still_count = static_count = 0
                prev_kp = None

            # ── cooldown
            if cooldown > 0:
                cooldown -= 1
                continue

            if hand:
                lost_hand_frames = 0
                count_frame += 1
                kp = extract_keypoints(results)

                if count_frame > 0:
                    kp_seq.append(kp)

                    still = is_still(kp, prev_kp)

                    if still:
                        still_count += 1
                    else:
                        still_count = 0

                    # ── RF
                    if rf_available and len(kp_seq) <= RF_ACTIVE_MAX_LEN and still:
                        static_count += 1
                        rf_model = rf2 if count_active_hands(kp) == 2 else rf1

                        if rf_model:
                            proba = rf_model["model"].predict_proba([kp])[0]
                            top = np.max(proba)
                            idx = np.argmax(proba)
                            letter = rf_model["classes"][idx]

                            if top > 0.55:
                                if letter == rf_last_letter:
                                    rf_confirm += 1
                                else:
                                    rf_last_letter = letter
                                    rf_confirm = 1

                                if rf_confirm >= RF_CONFIRM_FRAMES:
                                    sentence.append(letter)
                                    kp_seq = []
                                    cooldown = COOLDOWN_FRAMES
                                    prev_kp = None
                                    continue

                    # ── LSTM
                    if still_count >= STILLNESS_FRAMES and len(kp_seq) >= MIN_FRAMES:
                        seq = kp_seq[:-still_count] if still_count < len(kp_seq) else kp_seq
                        seq = normalize(seq)

                        x = tf.constant([seq], dtype=tf.float32)
                        res = predict_fn(x).numpy()[0]

                        top1 = np.max(res)
                        idx = np.argmax(res)
                        word = word_ids[idx]

                        if top1 > THRESHOLD:
                            sentence.append(word)

                        kp_seq = []
                        cooldown = COOLDOWN_FRAMES
                        prev_kp = None
                        continue

                prev_kp = kp

            else:
                if len(kp_seq) >= MIN_FRAMES:
                    seq = normalize(kp_seq)

                    x = tf.constant([seq], dtype=tf.float32)
                    res = predict_fn(x).numpy()[0]

                    if np.max(res) > THRESHOLD:
                        sentence.append(word_ids[np.argmax(res)])

                kp_seq = []
                prev_kp = None

    cap.release()
    return sentence