import streamlit as st
from gtts import gTTS
import tempfile
from datetime import date, datetime
import sqlite3
import io
import html
import hashlib
import hmac
import secrets
import re

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AgriCare AI",
    page_icon="🌱",
    layout="wide"
)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_NAME = "A2H0H0R1/mobilenet_v2_1.0_224-plant-disease"


@st.cache_resource
def load_ai_model():
    """
    Load the pretrained MobileNetV2 plant disease model.

    The model contains 38 plant disease / healthy classes.
    Streamlit caches the model after the first load.
    """

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    return processor, model


# =========================================================
# IMAGE HANDLING
# =========================================================

def read_uploaded_image(uploaded_file):
    """
    Safely read an uploaded image.

    Supports:
    JPG
    JPEG
    PNG
    WEBP

    The image is always converted to RGB before AI inference.
    """

    if uploaded_file is None:
        return None

    try:

        image_bytes = uploaded_file.getvalue()

        if not image_bytes:
            raise ValueError("The uploaded file is empty.")

        image = Image.open(
            io.BytesIO(image_bytes)
        )

        # Force PIL to completely load the image.
        image.load()

        # Correct phone-camera orientation when EXIF data is present.
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        detected_format = image.format or "Unknown"

        # Convert every supported format/mode to RGB.
        image = image.convert("RGB")

        # Store the detected format on the image object for display/debugging.
        image.info["detected_format"] = detected_format

        return image

    except UnidentifiedImageError:

        raise ValueError(
            "This file is not a valid image. "
            "Please upload JPG, JPEG, PNG or WEBP."
        )

    except Exception as e:

        raise ValueError(
            f"Could not read the image: {e}"
        )


def image_is_reasonable(image):
    """Validate image size and estimate whether it is useful for leaf AI."""
    if image is None:
        return False, "No image selected."
    width, height = image.size
    if width < 224 or height < 224:
        return False, "Image is too small. Please upload a clear leaf photo (at least 224×224)."

    # A simple sharpness check helps prevent blurry photos from producing misleading
    # high-confidence model predictions. This does not diagnose disease.
    try:
        gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < 25:
            return False, "Image is too blurry for reliable AI detection. Please take a sharper close-up photo of the leaf."
    except Exception:
        pass
    return True, ""


def assess_image_quality(image):
    """Return transparent quality indicators used by the AI result screen."""
    try:
        rgb = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(gray.mean())
        if sharpness < 25:
            quality = "Poor — blurry"
        elif sharpness < 80:
            quality = "Fair — retake if possible"
        elif brightness < 35 or brightness > 225:
            quality = "Fair — lighting may affect detection"
        else:
            quality = "Good"
        return {"sharpness": round(sharpness, 1), "brightness": round(brightness, 1), "quality": quality}
    except Exception:
        return {"sharpness": None, "brightness": None, "quality": "Unknown"}


# =========================================================
# RUN AI PREDICTION
# =========================================================

def model_crop_name(crop):
    """Map the app crop name to the crop names used by the model."""
    mapping = {
        "Maize": "Maize",
        "Tomato": "Tomato",
        "Chilli": "Chilli",
        "Potato": "Potato",
    }
    return mapping.get(crop)


def run_prediction(image, selected_crop=None):
    """Run a more robust crop-aware prediction with simple test-time augmentation.

    The model is evaluated on the original image and a horizontally flipped copy;
    their probabilities are averaged. For supported crops, unrelated crop classes
    are removed before the final decision. A consistency score is also returned so
    the UI can warn when the model is unstable instead of presenting a false sense
    of certainty.
    """
    processor, model = load_ai_model()
    if not isinstance(image, Image.Image):
        image = read_uploaded_image(image)
    image = image.convert("RGB")

    quality = assess_image_quality(image)
    augmented = [image, ImageOps.mirror(image)]
    probability_list = []
    with torch.no_grad():
        for img in augmented:
            inputs = processor(images=img, return_tensors="pt")
            outputs = model(**inputs)
            probability_list.append(torch.softmax(outputs.logits, dim=-1)[0])

    probabilities = torch.stack(probability_list).mean(dim=0)
    consistency = float(1.0 - torch.mean(torch.abs(probability_list[0] - probability_list[1])).item())
    consistency = max(0.0, min(1.0, consistency))

    label_map = model.config.id2label
    labels = [str(label_map.get(i, label_map.get(str(i), str(i)))) for i in range(len(probabilities))]

    raw_index = int(torch.argmax(probabilities).item())
    raw_label = labels[raw_index]
    raw_confidence = float(probabilities[raw_index].item() * 100)
    raw_crop = detect_crop_from_label(raw_label)

    target_crop = model_crop_name(selected_crop) if selected_crop else None
    allowed = []
    if target_crop:
        for i, label in enumerate(labels):
            if detect_crop_from_label(label).lower() == target_crop.lower():
                allowed.append(i)

    if allowed:
        masked = torch.full_like(probabilities, float("-inf"))
        idx_tensor = torch.tensor(allowed, dtype=torch.long)
        masked[idx_tensor] = probabilities[idx_tensor]
        selected_probabilities = torch.softmax(masked, dim=-1)
        confidence_tensor, selected_index_tensor = torch.max(selected_probabilities, dim=0)
        predicted_index = int(selected_index_tensor.item())
        confidence = float(confidence_tensor.item() * 100)
        crop_gated = True
    else:
        selected_probabilities = probabilities
        confidence_tensor, selected_index_tensor = torch.max(probabilities, dim=0)
        predicted_index = int(selected_index_tensor.item())
        confidence = float(confidence_tensor.item() * 100)
        crop_gated = False

    predicted_label = labels[predicted_index]

    # Reliability is deliberately separate from raw classifier confidence.
    # A high softmax score alone does not prove a field diagnosis.
    reliability = confidence
    if consistency < 0.70:
        reliability = min(reliability, 55)
    elif consistency < 0.85:
        reliability = min(reliability, 70)
    if quality["quality"].startswith("Poor"):
        reliability = min(reliability, 45)
    elif quality["quality"].startswith("Fair"):
        reliability = min(reliability, 70)

    top_k = min(5, len(probabilities))
    if crop_gated:
        top_values, top_indices = torch.topk(selected_probabilities, k=min(5, len(allowed)))
    else:
        top_values, top_indices = torch.topk(probabilities, k=top_k)

    top_predictions = []
    for value, index in zip(top_values.tolist(), top_indices.tolist()):
        top_predictions.append({"label": labels[int(index)], "confidence": round(float(value) * 100, 2)})

    return {
        "label": predicted_label,
        "confidence": round(confidence),
        "reliability": round(reliability),
        "consistency": round(consistency * 100),
        "image_quality": quality,
        "top_predictions": top_predictions,
        "raw_label": raw_label,
        "raw_confidence": round(raw_confidence),
        "raw_crop": raw_crop,
        "crop_gated": crop_gated,
        "selected_crop": selected_crop or "Not specified",
    }


# =========================================================
# LOCAL HISTORY DATABASE
# =========================================================

# =========================================================
# LOGIN / SIGNUP
# =========================================================

def init_users_table():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            display_name TEXT NOT NULL,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def hash_password(password, salt=None):
    if salt is None: salt = secrets.token_bytes(16)
    h = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200000)
    return salt.hex(), h.hex()

def verify_password(password, salt_hex, stored_hash):
    salt = bytes.fromhex(salt_hex)
    _, h = hash_password(password, salt)
    return hmac.compare_digest(h, stored_hash)

def username_is_valid(username):
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username.strip()))

def create_user(display_name, username, password):
    salt, password_hash = hash_password(password)
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (display_name, username, password_hash, salt, created_at) VALUES (?, ?, ?, ?, ?)",
                       (display_name.strip(), username.strip().lower(), password_hash, salt, datetime.now().isoformat(timespec="seconds")))
        conn.commit(); return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "Username already exists. Please choose another username."
    finally: conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_FILE); cursor = conn.cursor()
    cursor.execute("SELECT id, display_name, username, password_hash, salt FROM users WHERE username = ?", (username.strip().lower(),))
    user = cursor.fetchone(); conn.close()
    if not user: return None
    user_id, display_name, saved_username, password_hash, salt = user
    if verify_password(password, salt, password_hash):
        return {"id": user_id, "display_name": display_name, "username": saved_username}
    return None

def logout_user():
    st.session_state.authenticated = False; st.session_state.guest_mode = False
    st.session_state.user_id = None; st.session_state.username = ""; st.session_state.display_name = ""
    st.session_state.quick_page = "Home"

def render_login_signup():
    st.markdown("<div style='text-align:center; font-size:48px;'>🌱</div>", unsafe_allow_html=True)
    st.title("Welcome to AgriCare AI")
    st.markdown("<div style='text-align:center;'>Login or create an account to use AgriCare AI.</div>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 2, 1])
    with center:
        login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])
        with login_tab:
            username = st.text_input("Username", key="login_username")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("🔐 Login", use_container_width=True, type="primary", key="login_button"):
                if not username.strip() or not password: st.warning("Please enter your username and password.")
                else:
                    user = authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True; st.session_state.guest_mode = False
                        st.session_state.user_id = user["id"]; st.session_state.username = user["username"]; st.session_state.display_name = user["display_name"]
                        st.session_state.quick_page = "Home"; st.rerun()
                    else: st.error("Invalid username or password.")
        with signup_tab:
            display_name = st.text_input("Full Name", key="signup_name")
            username = st.text_input("Create Username", key="signup_username")
            password = st.text_input("Create Password", type="password", key="signup_password")
            confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")
            if st.button("📝 Create Account", use_container_width=True, type="primary", key="signup_button"):
                if not display_name.strip() or not username.strip() or not password: st.warning("Please fill in all fields.")
                elif not username_is_valid(username): st.warning("Username must be 3–40 characters and use only letters, numbers, ., _ or -.")
                elif len(password) < 6: st.warning("Password must contain at least 6 characters.")
                elif password != confirm_password: st.error("Passwords do not match.")
                else:
                    created, message = create_user(display_name, username, password)
                    if created:
                        user = authenticate_user(username, password)
                        if user:
                            st.session_state.authenticated = True
                            st.session_state.guest_mode = False
                            st.session_state.user_id = user["id"]
                            st.session_state.username = user["username"]
                            st.session_state.display_name = user["display_name"]
                            st.session_state.quick_page = "Home"
                            st.rerun()
                        else:
                            st.success("Account created successfully. Please use the Login tab.")
                    else:
                        st.error(message)
        st.markdown("---")
        st.subheader("🌾 Offline Activity")
        st.caption("You can open Offline Activity without creating an account.")
        if st.button("📚 Open Offline Activity", use_container_width=True, key="offline_from_auth"):
            st.session_state.guest_mode = True; st.session_state.quick_page = "Offline Activity"; st.rerun()

DB_FILE = "agricare_history.db"


def init_database():

    conn = sqlite3.connect(
        DB_FILE
    )

    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            crop TEXT NOT NULL,
            crop_age INTEGER,
            growth_stage TEXT,
            soil TEXT,
            water TEXT,
            season TEXT,
            health_score INTEGER,
            confidence INTEGER,
            disease TEXT,
            severity TEXT,
            management TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_scan(data):

    conn = sqlite3.connect(
        DB_FILE
    )

    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO scans (
            scan_date,
            crop,
            crop_age,
            growth_stage,
            soil,
            water,
            season,
            health_score,
            confidence,
            disease,
            severity,
            management
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        data["crop"],
        data["crop_age"],
        data["growth_stage"],
        data["soil"],
        data["water"],
        data["season"],
        data["health_score"],
        data["confidence"],
        data["disease"],
        data["severity"],
        data["management"]
    ))

    conn.commit()
    conn.close()


def get_scans():

    conn = sqlite3.connect(
        DB_FILE
    )

    rows = conn.execute("""
        SELECT
            id,
            scan_date,
            crop,
            crop_age,
            growth_stage,
            soil,
            water,
            season,
            health_score,
            confidence,
            disease,
            severity,
            management
        FROM scans
        ORDER BY id DESC
    """).fetchall()

    conn.close()

    return rows


init_database()
init_users_table()

if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "guest_mode" not in st.session_state: st.session_state.guest_mode = False
if "user_id" not in st.session_state: st.session_state.user_id = None
if "username" not in st.session_state: st.session_state.username = ""
if "display_name" not in st.session_state: st.session_state.display_name = ""


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = {}

if "scan_saved" not in st.session_state:
    st.session_state.scan_saved = False

if "quick_page" not in st.session_state:
    st.session_state.quick_page = "Home"

if "language" not in st.session_state:
    st.session_state.language = "English"

if "voice_audio_path" not in st.session_state:
    st.session_state.voice_audio_path = None

if "voice_language_used" not in st.session_state:
    st.session_state.voice_language_used = "English"

if "last_image_bytes" not in st.session_state:
    st.session_state.last_image_bytes = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #f5faf6;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.hero {
    background: linear-gradient(135deg, #1b7f3a, #4caf50);
    padding: 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 18px;
}

.card {
    background: linear-gradient(145deg, #f7fff8, #e8f7ec);
    padding: 25px;
    border-radius: 20px;
    margin-bottom: 20px;
    border: 1px solid #c5e5cc;
    box-shadow: 0px 8px 20px rgba(27,127,58,0.08);
}

.feature-card {
    background: linear-gradient(145deg, #f7fff8, #e4f7e9);
    padding: 22px;
    border-radius: 20px;
    min-height: 170px;
    border: 1px solid #bfe3c8;
    box-shadow: 0px 8px 22px rgba(27,127,58,0.10);
}

.metric-card {
    background: linear-gradient(145deg, #ffffff, #edf9f0);
    padding: 20px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #cce7d2;
    box-shadow: 0 6px 18px rgba(27,127,58,0.07);
}

.metric-number {
    font-size: 32px;
    font-weight: bold;
    color: #1b7f3a;
}

.small-text {
    color: #666;
    font-size: 14px;
}

.success-box {
    background-color: #e8f5e9;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #2e7d32;
}

.warning-box {
    background-color: #fff8e1;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #f9a825;
}

.danger-box {
    background-color: #ffebee;
    padding: 18px;
    border-radius: 14px;
    border-left: 5px solid #c62828;
}

.stButton > button {
    width: 100%;
    min-height: 58px;
    border-radius: 16px;
    border: 1px solid #b8dfc2;
    background: linear-gradient(135deg, #e8f8ed, #ccefd6);
    color: #145c2b;
    font-weight: 700;
    font-size: 16px;
    box-shadow: 0 5px 14px rgba(27, 127, 58, 0.10);
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1b7f3a, #4caf50);
    color: white;
    border-color: #1b7f3a;
    transform: translateY(-2px);
    box-shadow: 0 8px 18px rgba(27, 127, 58, 0.22);
}

.stButton > button:focus {
    border-color: #1b7f3a;
}

button[kind="primary"] {
    background: linear-gradient(135deg, #166534, #22a447) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 6px 16px rgba(22, 101, 52, 0.20) !important;
}

button[kind="primary"]:hover {
    background: linear-gradient(135deg, #0f4f28, #1b7f3a) !important;
    transform: translateY(-2px);
}

.quick-action-card {
    padding: 18px 18px 8px 18px;
    border-radius: 20px;
    margin-bottom: 8px;
    border: 1px solid #d8eadc;
    background: linear-gradient(145deg, #ffffff, #f0faf2);
    box-shadow: 0 6px 18px rgba(27, 127, 58, 0.08);
    text-align: center;
}

.quick-action-card .qa-icon {
    font-size: 30px;
    margin-bottom: 4px;
}

.quick-action-card .qa-title {
    font-size: 17px;
    font-weight: 800;
    color: #145c2b;
}

.quick-action-card .qa-text {
    font-size: 13px;
    color: #5f7165;
    margin-top: 3px;
    margin-bottom: 5px;
}

.ai-result {
    background: linear-gradient(145deg, #f1fff4, #dff5e5);
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #b7dfc1;
    box-shadow: 0 8px 22px rgba(27,127,58,0.10);
}

.ai-label {
    font-size: 14px;
    color: #587063;
    font-weight: 600;
}

.ai-value {
    font-size: 22px;
    color: #145c2b;
    font-weight: 800;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MULTILINGUAL UI
# =========================================================

TRANSLATIONS = {

    "English": {

        "navigation": "Navigation",
        "language": "🌐 Language",
        "smart": "Smart Farming Assistant",

        "home": "🏠 Home",
        "doctor": "🔬 AI Plant Doctor",
        "history": "📋 History",
        "new_farming": "🌾 New Farming",
        "offline": "📴 Offline Activity",
        "chat": "💬 AI Farming Chat",

        "welcome": "👋 Welcome Farmer",
        "protect": "Protect your crops, understand plant health and make better farming decisions with AI.",

        "main_features": "🚀 Main Features",
        "quick_actions": "⚡ Quick Actions",
        "quick_caption": "Start the most important farming tasks in one tap",

        "doctor_card": "🔬 AI Plant Doctor",
        "doctor_desc": "Upload or capture a crop image and analyze possible diseases and plant health problems.",

        "farm_card": "🌾 New Farming",
        "farm_desc": "Create a farming plan based on crop, soil, water, season and other farm information.",

        "chat_card": "💬 AI Farming Chat",
        "chat_desc": "Ask farming-related questions and get simple AI-based guidance.",

        "scan": "🔬 Crop Health Scan",
        "scan_desc": "Detect diseases & plant problems",

        "plan": "🌾 Plan a New Farm",
        "plan_desc": "Create your personalized farm plan",

        "ask": "🤖 Ask AgriCare AI",
        "ask_desc": "Get instant farming guidance",

        "supported": "Supported Crops",
        "detection": "AI Disease Detection",
        "languages": "Languages",
        "assistant": "Farming Assistant",

        "doctor_title": "🔬 AI Plant Doctor",
        "doctor_intro": "Upload a crop/leaf image or use your camera to check possible crop diseases and plant health.",

        "crop_info": "🌾 Crop Information",
        "select_crop": "Select Crop",
        "age": "Crop Age (days)",
        "growth": "Growth Stage",
        "soil": "Soil Type",
        "water": "Water Availability",
        "season": "Season",

        "image": "📷 Crop Image",
        "source": "Choose image source",
        "upload": "📤 Upload Image",
        "camera": "📸 Camera",

        "upload_prompt": "Upload crop or leaf image",
        "camera_prompt": "Take a photo of the crop leaf",
        "selected": "Selected Crop Image",

        "analyze": "🔍 Analyze Crop",
        "need_image": "⚠️ Please upload an image or take a photo first.",

        "analyzing": "🧠 AI is analyzing the image...",
        "model_error": "AI model could not analyze this image.",

        "analyzed": "✅ Image analyzed successfully!",
        "saved": "💾 This scan has been saved automatically to History.",

        "result": "🧠 AI Analysis Result",
        "health": "🌿 Health Score",
        "confidence": "🎯 Confidence",
        "disease": "🦠 Disease",
        "severity": "⚠️ Severity",

        "details": "📊 Detection Details",
        "selected_crop": "Selected Crop:",
        "crop_age": "Crop Age:",
        "growth_stage": "Growth Stage:",
        "water_availability": "Water Availability:",

        "ai_crop": "AI Detected Crop:",
        "problem": "AI Detected Problem:",
        "disease_conf": "Disease Confidence:",
        "crop_match": "Crop Match:",

        "management": "🩺 Recommended Management",
        "prevention": "🛡️ Prevention",

        "voice": "🔊 Voice Assistance",
        "voice_lang": "Choose Voice Language",
        "generate": "🎧 Generate Voice",
        "creating": "🔊 Creating voice...",
        "voice_ok": "✅ Voice generated successfully!",
        "audio": "### 🎧 AI Result Audio",
        "play": "▶️ Press the play button to listen to the AI result.",

        "feedback": "⭐ Farmer Feedback",
        "useful": "Was this result useful?",
        "rating": "Accuracy Rating",
        "comment": "Additional Feedback",
        "submit": "Submit Feedback",
        "thanks": "🙏 Thank you! Your feedback helps improve AgriCare AI.",

        "history_title": "📋 Farming History",
        "history_intro": "View previous crop scans, farming plans and AI questions.",

        "new_title": "🌾 New Farming",
        "farm_info": "🌱 Farm Information",

        "offline_title": "📴 Offline Activity",

        "chat_title": "💬 AI Farming Chat",
        "chat_intro": "Ask AgriCare AI questions about crops, soil, irrigation, pests and diseases.",

        "examples": "💡 Example Questions"
    },

    "తెలుగు": {

        "navigation": "నావిగేషన్",
        "language": "🌐 భాష",
        "smart": "స్మార్ట్ వ్యవసాయ సహాయకుడు",

        "home": "🏠 హోమ్",
        "doctor": "🔬 AI మొక్కల వైద్యుడు",
        "history": "📋 చరిత్ర",
        "new_farming": "🌾 కొత్త సాగు",
        "offline": "📴 ఆఫ్‌లైన్ సమాచారం",
        "chat": "💬 AI వ్యవసాయ చాట్",

        "welcome": "👋 రైతుకు స్వాగతం",
        "protect": "మీ పంటలను రక్షించండి, మొక్కల ఆరోగ్యాన్ని తెలుసుకోండి మరియు AI సహాయంతో మంచి వ్యవసాయ నిర్ణయాలు తీసుకోండి.",

        "main_features": "🚀 ప్రధాన ఫీచర్లు",
        "quick_actions": "⚡ త్వరిత చర్యలు",
        "quick_caption": "ముఖ్యమైన వ్యవసాయ పనులను ఒక ట్యాప్‌లో ప్రారంభించండి",

        "doctor_card": "🔬 AI మొక్కల వైద్యుడు",
        "doctor_desc": "పంట చిత్రాన్ని అప్‌లోడ్ చేసి వ్యాధులు మరియు మొక్కల ఆరోగ్య సమస్యలను గుర్తించండి.",

        "farm_card": "🌾 కొత్త సాగు",
        "farm_desc": "పంట, నేల, నీరు, సీజన్ మరియు ఇతర వివరాలతో సాగు ప్రణాళిక రూపొందించండి.",

        "chat_card": "💬 AI వ్యవసాయ చాట్",
        "chat_desc": "వ్యవసాయ ప్రశ్నలు అడిగి సులభమైన AI మార్గదర్శకత్వం పొందండి.",

        "scan": "🔬 పంట ఆరోగ్య స్కాన్",
        "scan_desc": "వ్యాధులు మరియు మొక్క సమస్యలను గుర్తించండి",

        "plan": "🌾 కొత్త సాగు ప్రణాళిక",
        "plan_desc": "మీ కోసం సాగు ప్రణాళిక రూపొందించండి",

        "ask": "🤖 AgriCare AIని అడగండి",
        "ask_desc": "తక్షణ వ్యవసాయ మార్గదర్శకత్వం పొందండి",

        "supported": "మద్దతు ఉన్న పంటలు",
        "detection": "AI వ్యాధి గుర్తింపు",
        "languages": "భాషలు",
        "assistant": "వ్యవసాయ సహాయకుడు",

        "doctor_title": "🔬 AI మొక్కల వైద్యుడు",
        "doctor_intro": "పంట/ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా కెమెరాతో చిత్రాన్ని తీసి పంట వ్యాధులు మరియు ఆరోగ్యాన్ని పరిశీలించండి.",

        "crop_info": "🌾 పంట సమాచారం",
        "select_crop": "పంటను ఎంచుకోండి",
        "age": "పంట వయస్సు (రోజులు)",
        "growth": "పెరుగుదల దశ",
        "soil": "నేల రకం",
        "water": "నీటి లభ్యత",
        "season": "సీజన్",

        "image": "📷 పంట చిత్రం",
        "source": "చిత్రం మూలాన్ని ఎంచుకోండి",
        "upload": "📤 చిత్రం అప్‌లోడ్",
        "camera": "📸 కెమెరా",

        "upload_prompt": "పంట లేదా ఆకు చిత్రాన్ని అప్‌లోడ్ చేయండి",
        "camera_prompt": "పంట ఆకుకు ఫోటో తీయండి",
        "selected": "ఎంచుకున్న పంట చిత్రం",

        "analyze": "🔍 పంటను విశ్లేషించండి",
        "need_image": "⚠️ ముందుగా చిత్రాన్ని అప్‌లోడ్ చేయండి లేదా ఫోటో తీయండి.",

        "analyzing": "🧠 AI చిత్రాన్ని విశ్లేషిస్తోంది...",
        "model_error": "AI మోడల్ ఈ చిత్రాన్ని విశ్లేషించలేకపోయింది.",

        "analyzed": "✅ చిత్రం విజయవంతంగా విశ్లేషించబడింది!",
        "saved": "💾 ఈ స్కాన్ చరిత్రలో ఆటోమేటిక్‌గా సేవ్ చేయబడింది.",

        "result": "🧠 AI విశ్లేషణ ఫలితం",
        "health": "🌿 ఆరోగ్య స్కోర్",
        "confidence": "🎯 నమ్మక స్థాయి",
        "disease": "🦠 వ్యాధి",
        "severity": "⚠️ తీవ్రత",

        "details": "📊 గుర్తింపు వివరాలు",
        "selected_crop": "ఎంచుకున్న పంట:",
        "crop_age": "పంట వయస్సు:",
        "growth_stage": "పెరుగుదల దశ:",
        "water_availability": "నీటి లభ్యత:",

        "ai_crop": "AI గుర్తించిన పంట:",
        "problem": "AI గుర్తించిన సమస్య:",
        "disease_conf": "వ్యాధి నమ్మక స్థాయి:",
        "crop_match": "పంట సరిపోలిక:",

        "management": "🩺 సూచించిన నిర్వహణ",
        "prevention": "🛡️ నివారణ",

        "voice": "🔊 వాయిస్ సహాయం",
        "voice_lang": "వాయిస్ భాషను ఎంచుకోండి",
        "generate": "🎧 వాయిస్ రూపొందించండి",
        "creating": "🔊 వాయిస్ రూపొందుతోంది...",
        "voice_ok": "✅ వాయిస్ విజయవంతంగా రూపొందించబడింది!",
        "audio": "### 🎧 AI ఫలిత ఆడియో",
        "play": "▶️ AI ఫలితాన్ని వినడానికి ప్లే బటన్ నొక్కండి.",

        "feedback": "⭐ రైతు అభిప్రాయం",
        "useful": "ఈ ఫలితం ఉపయోగకరంగా ఉందా?",
        "rating": "ఖచ్చితత్వ రేటింగ్",
        "comment": "అదనపు అభిప్రాయం",
        "submit": "అభిప్రాయాన్ని పంపండి",
        "thanks": "🙏 ధన్యవాదాలు! మీ అభిప్రాయం AgriCare AIని మెరుగుపరచడంలో సహాయపడుతుంది.",

        "history_title": "📋 వ్యవసాయ చరిత్ర",
        "history_intro": "గత పంట స్కాన్లు, సాగు ప్రణాళికలు మరియు AI ప్రశ్నలను చూడండి.",

        "new_title": "🌾 కొత్త సాగు",
        "farm_info": "🌱 పొలం సమాచారం",

        "offline_title": "📴 ఆఫ్‌లైన్ సమాచారం",

        "chat_title": "💬 AI వ్యవసాయ చాట్",
        "chat_intro": "పంటలు, నేల, నీరు, తెగుళ్లు మరియు వ్యాధుల గురించి AgriCare AIని అడగండి.",

        "examples": "💡 ఉదాహరణ ప్రశ్నలు"
    },

    "हिंदी": {

        "navigation": "नेविगेशन",
        "language": "🌐 भाषा",
        "smart": "स्मार्ट कृषि सहायक",

        "home": "🏠 होम",
        "doctor": "🔬 AI प्लांट डॉक्टर",
        "history": "📋 इतिहास",
        "new_farming": "🌾 नई खेती",
        "offline": "📴 ऑफलाइन जानकारी",
        "chat": "💬 AI कृषि चैट",

        "welcome": "👋 किसान का स्वागत है",
        "protect": "अपनी फसलों की सुरक्षा करें, पौधों का स्वास्थ्य समझें और AI की मदद से बेहतर कृषि निर्णय लें।",

        "main_features": "🚀 मुख्य फीचर्स",
        "quick_actions": "⚡ त्वरित कार्य",
        "quick_caption": "महत्वपूर्ण कृषि कार्य एक टैप में शुरू करें",

        "doctor_card": "🔬 AI प्लांट डॉक्टर",
        "doctor_desc": "फसल की तस्वीर अपलोड करके रोग और पौधों की स्वास्थ्य समस्याओं का विश्लेषण करें।",

        "farm_card": "🌾 नई खेती",
        "farm_desc": "फसल, मिट्टी, पानी, मौसम और अन्य जानकारी के आधार पर खेती की योजना बनाएं।",

        "chat_card": "💬 AI कृषि चैट",
        "chat_desc": "कृषि से जुड़े सवाल पूछें और आसान AI मार्गदर्शन पाएं।",

        "scan": "🔬 फसल स्वास्थ्य स्कैन",
        "scan_desc": "रोग और पौधों की समस्याएं पहचानें",

        "plan": "🌾 नई खेती की योजना",
        "plan_desc": "अपनी व्यक्तिगत खेती योजना बनाएं",

        "ask": "🤖 AgriCare AI से पूछें",
        "ask_desc": "तुरंत कृषि मार्गदर्शन पाएं",

        "supported": "समर्थित फसलें",
        "detection": "AI रोग पहचान",
        "languages": "भाषाएं",
        "assistant": "कृषि सहायक",

        "doctor_title": "🔬 AI प्लांट डॉक्टर",
        "doctor_intro": "फसल/पत्ती की तस्वीर अपलोड करें या कैमरे से फोटो लें और संभावित रोगों व पौधों के स्वास्थ्य की जांच करें।",

        "crop_info": "🌾 फसल की जानकारी",
        "select_crop": "फसल चुनें",
        "age": "फसल की आयु (दिन)",
        "growth": "विकास चरण",
        "soil": "मिट्टी का प्रकार",
        "water": "पानी की उपलब्धता",
        "season": "मौसम",

        "image": "📷 फसल की तस्वीर",
        "source": "तस्वीर का स्रोत चुनें",
        "upload": "📤 तस्वीर अपलोड करें",
        "camera": "📸 कैमरा",

        "upload_prompt": "फसल या पत्ती की तस्वीर अपलोड करें",
        "camera_prompt": "फसल की पत्ती की फोटो लें",
        "selected": "चयनित फसल तस्वीर",

        "analyze": "🔍 फसल का विश्लेषण करें",
        "need_image": "⚠️ पहले तस्वीर अपलोड करें या फोटो लें।",

        "analyzing": "🧠 AI तस्वीर का विश्लेषण कर रहा है...",
        "model_error": "AI मॉडल इस तस्वीर का विश्लेषण नहीं कर सका।",

        "analyzed": "✅ तस्वीर का सफलतापूर्वक विश्लेषण हुआ!",
        "saved": "💾 यह स्कैन इतिहास में अपने आप सेव हो गया है।",

        "result": "🧠 AI विश्लेषण परिणाम",
        "health": "🌿 स्वास्थ्य स्कोर",
        "confidence": "🎯 विश्वास स्तर",
        "disease": "🦠 रोग",
        "severity": "⚠️ गंभीरता",

        "details": "📊 पहचान विवरण",
        "selected_crop": "चयनित फसल:",
        "crop_age": "फसल की आयु:",
        "growth_stage": "विकास चरण:",
        "water_availability": "पानी की उपलब्धता:",

        "ai_crop": "AI द्वारा पहचानी फसल:",
        "problem": "AI द्वारा पहचानी समस्या:",
        "disease_conf": "रोग विश्वास स्तर:",
        "crop_match": "फसल मिलान:",

        "management": "🩺 सुझाया गया प्रबंधन",
        "prevention": "🛡️ रोकथाम",

        "voice": "🔊 वॉयस सहायता",
        "voice_lang": "वॉयस भाषा चुनें",
        "generate": "🎧 वॉयस बनाएं",
        "creating": "🔊 वॉयस बनाई जा रही है...",
        "voice_ok": "✅ वॉयस सफलतापूर्वक बनाई गई!",
        "audio": "### 🎧 AI परिणाम ऑडियो",
        "play": "▶️ AI परिणाम सुनने के लिए प्ले बटन दबाएं।",

        "feedback": "⭐ किसान प्रतिक्रिया",
        "useful": "क्या यह परिणाम उपयोगी था?",
        "rating": "सटीकता रेटिंग",
        "comment": "अतिरिक्त प्रतिक्रिया",
        "submit": "प्रतिक्रिया भेजें",
        "thanks": "🙏 धन्यवाद! आपकी प्रतिक्रिया AgriCare AI को बेहतर बनाने में मदद करती है।",

        "history_title": "📋 कृषि इतिहास",
        "history_intro": "पिछले फसल स्कैन, खेती की योजनाएं और AI सवाल देखें।",

        "new_title": "🌾 नई खेती",
        "farm_info": "🌱 खेत की जानकारी",

        "offline_title": "📴 ऑफलाइन जानकारी",

        "chat_title": "💬 AI कृषि चैट",
        "chat_intro": "फसल, मिट्टी, सिंचाई, कीट और रोगों के बारे में AgriCare AI से पूछें।",

        "examples": "💡 उदाहरण प्रश्न"
    }
}


def tr(key):

    return TRANSLATIONS.get(
        st.session_state.language,
        TRANSLATIONS["English"]
    ).get(
        key,
        TRANSLATIONS["English"].get(
            key,
            key
        )
    )


# =========================================================
# DISEASE / CROP INFORMATION
# =========================================================

CROP_KEYWORDS = {

    "Tomato": ["tomato"],

    "Potato": ["potato"],

    "Maize": [
        "maize",
        "corn"
    ],

    "Rice": ["rice"],

    "Chilli": [
        "pepper",
        "chilli",
        "chili"
    ],

    "Grape": ["grape"],

    "Apple": ["apple"],

    "Peach": ["peach"],

    "Cherry": ["cherry"],

    "Strawberry": ["strawberry"],

    "Blueberry": ["blueberry"],

    "Raspberry": ["raspberry"],

    "Soybean": ["soybean"],

    "Squash": ["squash"],
}


DISEASE_GUIDANCE = {

    "healthy": {

        "name": "Healthy Plant",

        "management": [
            "Continue regular crop monitoring.",
            "Maintain balanced irrigation.",
            "Keep the field clean and well ventilated.",
            "Continue suitable nutrition according to local recommendations."
        ],

        "prevention": [
            "Inspect leaves regularly.",
            "Use healthy planting material.",
            "Avoid unnecessary leaf wetness.",
            "Monitor early symptoms."
        ]
    },


    "early blight": {

        "name": "Early Blight",

        "management": [
            "Remove severely affected leaves and plant debris.",
            "Improve spacing and air circulation.",
            "Avoid unnecessary overhead irrigation.",
            "Monitor nearby plants for similar lesions.",
            "Use locally recommended disease-management practices."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Rotate crops where practical.",
            "Remove infected plant debris.",
            "Avoid prolonged leaf wetness."
        ]
    },


    "late blight": {

        "name": "Late Blight",

        "management": [
            "Remove severely infected plant material.",
            "Improve field ventilation.",
            "Avoid prolonged leaf wetness.",
            "Monitor surrounding plants closely.",
            "Consult local agricultural guidance for approved fungicide options."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Avoid excessive moisture around foliage.",
            "Monitor frequently during favorable weather.",
            "Remove infected debris."
        ]
    },


    "leaf spot": {

        "name": "Leaf Spot",

        "management": [
            "Remove severely infected leaves.",
            "Maintain proper spacing between plants.",
            "Avoid unnecessary water on leaves.",
            "Keep infected plant material away from healthy plants.",
            "Monitor nearby plants."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Maintain field sanitation.",
            "Avoid overcrowding.",
            "Monitor crop health regularly."
        ]
    },


    "rust": {

        "name": "Rust",

        "management": [
            "Remove heavily affected leaves where practical.",
            "Improve air circulation.",
            "Avoid unnecessary leaf wetness.",
            "Monitor surrounding plants.",
            "Follow locally recommended disease-management practices."
        ],

        "prevention": [
            "Use resistant varieties when locally available.",
            "Maintain suitable plant spacing.",
            "Monitor early symptoms.",
            "Maintain field sanitation."
        ]
    },


    "mosaic": {

        "name": "Mosaic Disease",

        "management": [
            "Remove severely affected plants where appropriate.",
            "Control insect vectors according to local recommendations.",
            "Remove heavily infected plant material.",
            "Monitor nearby plants for symptoms."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Control vector insects.",
            "Remove infected plants promptly.",
            "Inspect new plants before introduction."
        ]
    },


    "bacterial": {

        "name": "Bacterial Disease",

        "management": [
            "Remove severely affected plant material.",
            "Avoid unnecessary handling of wet plants.",
            "Improve field sanitation.",
            "Avoid excessive leaf wetness.",
            "Consult local agricultural guidance for treatment options."
        ],

        "prevention": [
            "Use clean planting material.",
            "Maintain sanitation.",
            "Avoid working in wet foliage.",
            "Monitor crop regularly."
        ]
    },


    "powdery mildew": {

        "name": "Powdery Mildew",

        "management": [
            "Remove severely affected leaves where practical.",
            "Improve air circulation.",
            "Avoid excessive nitrogen application.",
            "Monitor new growth carefully.",
            "Follow locally recommended disease-management practices."
        ],

        "prevention": [
            "Maintain proper plant spacing.",
            "Use resistant varieties when available.",
            "Avoid excessive humidity around foliage.",
            "Monitor early symptoms."
        ]
    },


    "black rot": {

        "name": "Black Rot",

        "management": [
            "Remove severely affected plant material.",
            "Remove infected debris from the field.",
            "Improve field sanitation.",
            "Avoid prolonged leaf wetness.",
            "Follow local disease-management recommendations."
        ],

        "prevention": [
            "Use clean planting material.",
            "Maintain field sanitation.",
            "Avoid unnecessary leaf wetness.",
            "Monitor plants regularly."
        ]
    },


    "scab": {

        "name": "Scab",

        "management": [
            "Remove infected plant material where practical.",
            "Maintain field sanitation.",
            "Remove fallen infected leaves and debris.",
            "Monitor new growth.",
            "Follow locally recommended disease-management practices."
        ],

        "prevention": [
            "Use healthy planting material.",
            "Maintain sanitation.",
            "Monitor regularly.",
            "Follow local agricultural guidance."
        ]
    },


    "default": {

        "name": "Plant Disease / Stress",

        "management": [
            "Inspect the affected plant closely.",
            "Remove severely damaged leaves if appropriate.",
            "Maintain proper irrigation.",
            "Improve field sanitation and plant spacing.",
            "For treatment products, follow local agricultural recommendations."
        ],

        "prevention": [
            "Monitor crops regularly.",
            "Use healthy planting material.",
            "Maintain proper irrigation.",
            "Avoid overcrowding.",
            "Seek expert verification for uncertain cases."
        ]
    }
}


# =========================================================
# CROP-SPECIFIC MEDICINE & MANAGEMENT DATABASE
# =========================================================
# This database is intentionally active-ingredient based instead of brand based.
# Product names, formulations, label claims, doses, PHI and permitted crops vary
# by country/state/formulation. The app therefore tells the farmer to follow the
# current locally registered product label. ICAR/NRCIPM guidance is used as a
# reference for several entries.

CROP_DISEASE_GUIDANCE = {
    "Tomato": {
        "healthy": {
            "name": "Healthy Tomato", "problem": "No disease class was identified with the available model.",
            "medicine_type": "No medicine indicated", "active_ingredients": [],
            "treatment": ["No disease medicine is indicated from this scan.", "Continue regular scouting, balanced irrigation and good airflow."],
            "prevention": ["Use healthy seedlings.", "Remove diseased crop debris.", "Avoid prolonged leaf wetness."]
        },
        "early blight": {
            "name": "Tomato Early Blight", "problem": "Fungal disease producing dark target-like leaf spots, often on older leaves.",
            "medicine_type": "Fungicide", "active_ingredients": ["Chlorothalonil", "Mancozeb", "Copper oxychloride"],
            "treatment": ["Remove badly infected leaves where practical.", "Use a locally registered tomato fungicide containing an appropriate active ingredient such as chlorothalonil, mancozeb or copper oxychloride, only according to its current label.", "Rotate fungicide groups where the label permits."],
            "prevention": ["Maintain plant spacing and airflow.", "Avoid overhead irrigation.", "Remove infected debris after harvest."]
        },
        "late blight": {
            "name": "Tomato Late Blight", "problem": "A rapidly spreading disease favored by cool, wet conditions.",
            "medicine_type": "Fungicide", "active_ingredients": ["Mancozeb", "Metalaxyl + Mancozeb", "Cymoxanil + Mancozeb", "Chlorothalonil"],
            "treatment": ["Remove heavily infected material where practical.", "Use only a locally registered late-blight fungicide. ICAR advisories include copper oxychloride and metalaxyl 4% + mancozeb 64% for tomato disease management.", "Start protection early when disease risk is high and follow the label and resistance-management instructions."],
            "prevention": ["Improve airflow and drainage.", "Avoid prolonged leaf wetness.", "Scout frequently during cool and wet weather."]
        },
        "bacterial spot": {
            "name": "Tomato Bacterial Spot", "problem": "Bacterial lesions can appear as small dark or water-soaked spots on leaves and fruit.",
            "medicine_type": "Bactericide / protective treatment", "active_ingredients": ["Copper-based products where registered"],
            "treatment": ["Remove severely affected material where practical.", "Use only a locally registered bactericide for tomato bacterial spot.", "Do not mix products unless the label specifically permits the mixture."],
            "prevention": ["Use disease-free planting material.", "Avoid splash irrigation.", "Sanitize tools and avoid working wet foliage."]
        },
        "leaf mold": {
            "name": "Tomato Leaf Mold", "problem": "Leaf mold is favored by high humidity and poor airflow.",
            "medicine_type": "Fungicide", "active_ingredients": ["Use a locally registered tomato fungicide for leaf mold"],
            "treatment": ["Improve ventilation and reduce leaf wetness.", "Remove heavily infected leaves.", "Use only a locally registered fungicide after confirming the disease."],
            "prevention": ["Increase spacing and airflow.", "Avoid unnecessary overhead irrigation."]
        },
        "septoria leaf spot": {
            "name": "Tomato Septoria Leaf Spot", "problem": "Small circular leaf spots can enlarge and cause premature leaf loss.",
            "medicine_type": "Fungicide", "active_ingredients": ["Mancozeb", "Chlorothalonil where registered"],
            "treatment": ["Remove infected lower leaves and crop debris.", "Use a locally registered tomato fungicide according to the label.", "Avoid splashing soil onto foliage."],
            "prevention": ["Use clean seedlings.", "Rotate crops where practical.", "Maintain field sanitation."]
        },
        "target spot": {
            "name": "Tomato Target Spot", "problem": "Target-like concentric lesions may occur on tomato leaves and fruit.",
            "medicine_type": "Fungicide", "active_ingredients": ["Use a locally registered fungicide for target spot"],
            "treatment": ["Remove badly affected material.", "Use only a locally registered fungicide for confirmed target spot and rotate modes of action according to the label.", "Improve airflow and reduce leaf wetness."],
            "prevention": ["Maintain spacing and sanitation.", "Scout early during humid weather."]
        },
        "spider mites": {
            "name": "Tomato Spider Mites", "problem": "Mites can cause fine stippling, yellowing and webbing, especially on leaf undersides.",
            "medicine_type": "Miticide / acaricide", "active_ingredients": ["Fenpyroximate", "Spiromesifen", "Propargite where registered"],
            "treatment": ["Inspect leaf undersides before treatment.", "Conserve beneficial predators where possible.", "If treatment is required, use only an acaricide registered for tomato mites and follow its label."],
            "prevention": ["Avoid severe plant water stress.", "Scout leaf undersides regularly.", "Avoid unnecessary broad-spectrum insecticide use that can disrupt beneficial mites and insects."]
        },
        "yellow leaf curl": {
            "name": "Tomato Yellow Leaf Curl Virus", "problem": "Virus symptoms may include leaf curling, yellowing and stunting; whiteflies can spread the virus.",
            "medicine_type": "No curative medicine; vector management", "active_ingredients": ["No chemical cure for established virus", "Use only locally registered whitefly-control products when required"],
            "treatment": ["There is no curative pesticide that reverses an established viral infection.", "Remove severely affected plants where practical.", "Manage whitefly vectors using integrated pest management and only locally registered products when needed."],
            "prevention": ["Use healthy seedlings and resistant varieties where available.", "Monitor whiteflies early.", "Control weeds and volunteer hosts."]
        },
        "mosaic": {
            "name": "Tomato Mosaic Virus", "problem": "Mosaic viruses can cause mottling, distortion and reduced growth.",
            "medicine_type": "No curative medicine; sanitation and vector management", "active_ingredients": ["No chemical cure for established virus"],
            "treatment": ["There is no curative pesticide that restores a virus-infected plant.", "Remove severely infected plants where practical.", "Sanitize hands and tools and manage relevant insect vectors."],
            "prevention": ["Use certified healthy planting material.", "Control weeds and alternate hosts.", "Sanitize tools between plants."]
        }
    },

    "Potato": {
        "healthy": {
            "name": "Healthy Potato", "problem": "No disease class was identified with the available model.",
            "medicine_type": "No medicine indicated", "active_ingredients": [],
            "treatment": ["No disease medicine is indicated from this scan.", "Continue scouting and maintain good drainage."],
            "prevention": ["Use certified healthy seed tubers.", "Maintain field sanitation and drainage."]
        },
        "early blight": {
            "name": "Potato Early Blight", "problem": "Dark lesions with concentric rings often develop on older foliage.",
            "medicine_type": "Fungicide", "active_ingredients": ["Mancozeb", "Chlorothalonil where registered"],
            "treatment": ["Remove badly affected foliage where practical.", "Use a locally registered potato fungicide according to the current label.", "Rotate fungicide groups where permitted."],
            "prevention": ["Use healthy seed.", "Rotate crops where practical.", "Avoid prolonged leaf wetness and plant stress."]
        },
        "late blight": {
            "name": "Potato Late Blight", "problem": "Late blight can rapidly damage leaves and tubers under cool, wet conditions.",
            "medicine_type": "Fungicide", "active_ingredients": ["Mancozeb", "Cymoxanil", "Dimethomorph", "Fluopicolide + Propamocarb", "Chlorothalonil"],
            "treatment": ["Remove heavily infected foliage where feasible and manage infected tubers after harvest.", "Use only locally registered late-blight fungicides. ICAR sources report mancozeb, cymoxanil/dimethomorph-based products and fluopicolide + propamocarb programs as management options in appropriate settings.", "Follow the label, resistance-management guidance and pre-harvest interval."],
            "prevention": ["Use certified seed tubers.", "Improve drainage.", "Scout frequently during cool, wet weather."]
        }
    },

    "Maize": {
        "healthy": {
            "name": "Healthy Maize", "problem": "No disease class was identified with the available model.",
            "medicine_type": "No medicine indicated", "active_ingredients": [],
            "treatment": ["No disease medicine is indicated from this scan.", "Continue scouting for leaf diseases and insect damage."],
            "prevention": ["Use healthy seed.", "Maintain balanced nutrition and field sanitation."]
        },
        "northern leaf blight": {
            "name": "Maize Northern Leaf Blight", "problem": "Long cigar-shaped lesions can expand across maize leaves.",
            "medicine_type": "Fungicide", "active_ingredients": ["Use a locally registered maize fungicide for northern leaf blight"],
            "treatment": ["Confirm the disease before spraying.", "Use only a locally registered maize fungicide and follow its resistance-management and pre-harvest requirements.", "Do not spray when symptoms may be caused by nutrition or weather stress."],
            "prevention": ["Use tolerant/resistant varieties where available.", "Rotate crops and manage infected residue where practical."]
        },
        "gray leaf spot": {
            "name": "Maize Gray Leaf Spot", "problem": "Gray/tan rectangular lesions can expand along maize leaves, especially in humid conditions.",
            "medicine_type": "Fungicide", "active_ingredients": ["Azoxystrobin or other locally registered maize fungicide"],
            "treatment": ["Confirm the disease before chemical control.", "Use only a locally registered maize fungicide according to the label.", "Rotate modes of action where the label permits."],
            "prevention": ["Use tolerant varieties.", "Rotate crops.", "Manage infected residue and monitor lower leaves early."]
        },
        "cercospora": {
            "name": "Maize Cercospora Gray Leaf Spot", "problem": "Cercospora leaf spot can reduce green leaf area and yield when severe.",
            "medicine_type": "Fungicide", "active_ingredients": ["Use a locally registered maize fungicide for Cercospora leaf spot"],
            "treatment": ["Use chemical control only when the disease is confirmed and control is justified.", "Rotate fungicide modes of action according to the label.", "Monitor the lower canopy early."],
            "prevention": ["Use tolerant varieties.", "Rotate crops and manage infected residue."]
        },
        "common rust": {
            "name": "Maize Common Rust", "problem": "Small reddish-brown rust pustules can develop on maize leaves.",
            "medicine_type": "Fungicide", "active_ingredients": ["Azoxystrobin", "Propiconazole where registered"],
            "treatment": ["Confirm rust before treatment.", "Where control is justified, use a locally registered maize fungicide such as an approved strobilurin/triazole product according to the label.", "Follow all worker-protection and pre-harvest requirements."],
            "prevention": ["Use resistant varieties where available.", "Scout early and avoid severe crop stress."]
        }
    },

    "Chilli": {
        "healthy": {
            "name": "Healthy Chilli/Pepper", "problem": "No disease class was identified with the available model.",
            "medicine_type": "No medicine indicated", "active_ingredients": [],
            "treatment": ["No disease medicine is indicated from this scan.", "Continue scouting leaves, flowers and fruits."],
            "prevention": ["Use healthy seedlings.", "Control weeds and monitor insect vectors."]
        },
        "bacterial": {
            "name": "Chilli/Pepper Bacterial Spot", "problem": "Bacterial disease can cause dark or water-soaked lesions and fruit damage.",
            "medicine_type": "Bactericide / protective treatment", "active_ingredients": ["Copper-based products where registered"],
            "treatment": ["Remove severely infected material where practical.", "Use only a locally registered bactericide for the confirmed disease.", "Follow the label and do not mix products unless the label permits."],
            "prevention": ["Use disease-free seedlings.", "Avoid working wet foliage.", "Sanitize tools and reduce splash irrigation."]
        },
        "leaf spot": {
            "name": "Chilli Leaf Spot", "problem": "Leaf spots can have fungal, bacterial or environmental causes and need field confirmation.",
            "medicine_type": "Fungicide / protective treatment", "active_ingredients": ["Chlorothalonil where registered", "Copper-based products where registered"],
            "treatment": ["Remove severely affected leaves where practical.", "ICAR advisories list chlorothalonil for chilli leaf-spot management; use only a locally registered formulation and follow its current label.", "Improve airflow and avoid overhead irrigation."],
            "prevention": ["Use clean planting material.", "Maintain spacing and sanitation.", "Scout regularly during humid weather."]
        },
        "spider mites": {
            "name": "Chilli Mites", "problem": "Mites can cause curling, bronzing or distorted young leaves.",
            "medicine_type": "Miticide / acaricide", "active_ingredients": ["Fenpyroximate where registered", "Spiromesifen where registered"],
            "treatment": ["Inspect young leaves and leaf undersides.", "Use an acaricide only when mites are confirmed and the product is registered for chilli.", "Follow the current label and protect beneficial organisms."],
            "prevention": ["Avoid severe water stress.", "Scout young growth frequently."]
        },
        "yellow leaf curl": {
            "name": "Chilli Leaf Curl Virus", "problem": "Leaf curl is commonly associated with viral infection and insect vectors such as whiteflies.",
            "medicine_type": "No curative medicine; vector management", "active_ingredients": ["No chemical cure for established virus", "Use only locally registered vector-control products when required"],
            "treatment": ["There is no curative pesticide for an established viral infection.", "Remove severely affected plants where practical.", "Manage whitefly vectors through integrated pest management and locally registered products when necessary."],
            "prevention": ["Use healthy seedlings.", "Monitor vectors early.", "Remove weeds and volunteer hosts."]
        }
    }
}

# Backward-compatible aliases for the PlantVillage-style model labels.
CROP_DISEASE_GUIDANCE["Tomato"]["yellow leaf curl virus"] = CROP_DISEASE_GUIDANCE["Tomato"]["yellow leaf curl"]
CROP_DISEASE_GUIDANCE["Tomato"]["mosaic virus"] = CROP_DISEASE_GUIDANCE["Tomato"]["mosaic"]
CROP_DISEASE_GUIDANCE["Chilli"]["bacterial spot"] = CROP_DISEASE_GUIDANCE["Chilli"]["bacterial"]


def get_crop_disease_guidance(crop, predicted_label):
    normalized = normalize_label(predicted_label)
    crop_data = CROP_DISEASE_GUIDANCE.get(crop, {})
    if "healthy" in normalized:
        return crop_data.get("healthy", DISEASE_GUIDANCE["healthy"])

    # Check the most specific model labels first.
    disease_keys = [
        "yellow leaf curl virus", "yellow leaf curl", "mosaic virus", "mosaic",
        "bacterial spot", "leaf mold", "septoria leaf spot", "target spot",
        "spider mites", "northern leaf blight", "gray leaf spot", "cercospora",
        "common rust", "late blight", "early blight", "leaf spot", "rust", "bacterial"
    ]
    for key in disease_keys:
        if key in normalized and key in crop_data:
            return crop_data[key]

    return {
        "name": "Uncertain crop-specific result",
        "problem": "The image model did not provide a crop-specific treatment class with enough certainty.",
        "medicine_type": "Do not apply disease-specific medicine yet",
        "active_ingredients": [],
        "treatment": [
            "Do not apply a disease-specific chemical based only on this uncertain result.",
            "Retake 2–3 clear close-up photos of affected and healthy leaves.",
            "If symptoms are spreading or severe, confirm the diagnosis with a local agricultural officer or qualified agronomist before treatment."
        ],
        "prevention": ["Continue field scouting.", "Maintain sanitation and avoid unnecessary pesticide use."]
    }

# =========================================================
# LABEL HELPERS
# =========================================================

def normalize_label(label):

    return (
        str(label)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def get_guidance(predicted_label):

    normalized = normalize_label(
        predicted_label
    )

    if "healthy" in normalized:
        return DISEASE_GUIDANCE["healthy"]

    if "early blight" in normalized:
        return DISEASE_GUIDANCE["early blight"]

    if "late blight" in normalized:
        return DISEASE_GUIDANCE["late blight"]

    if (
        "leaf spot" in normalized
        or "leafspot" in normalized
        or "gray leaf spot" in normalized
    ):
        return DISEASE_GUIDANCE["leaf spot"]

    if "rust" in normalized:
        return DISEASE_GUIDANCE["rust"]

    if "mosaic" in normalized:
        return DISEASE_GUIDANCE["mosaic"]

    if "bacterial" in normalized:
        return DISEASE_GUIDANCE["bacterial"]

    if "powdery mildew" in normalized:
        return DISEASE_GUIDANCE["powdery mildew"]

    if "black rot" in normalized:
        return DISEASE_GUIDANCE["black rot"]

    if "scab" in normalized:
        return DISEASE_GUIDANCE["scab"]

    return DISEASE_GUIDANCE["default"]


def detect_crop_from_label(label):

    normalized = normalize_label(
        label
    )

    crop_order = [

        (
            "Tomato",
            ["tomato"]
        ),

        (
            "Potato",
            ["potato"]
        ),

        (
            "Maize",
            ["maize", "corn"]
        ),

        (
            "Rice",
            ["rice"]
        ),

        (
            "Chilli",
            ["pepper", "chilli", "chili"]
        ),

        (
            "Grape",
            ["grape"]
        ),

        (
            "Apple",
            ["apple"]
        ),

        (
            "Peach",
            ["peach"]
        ),

        (
            "Cherry",
            ["cherry"]
        ),

        (
            "Strawberry",
            ["strawberry"]
        ),

        (
            "Blueberry",
            ["blueberry"]
        ),

        (
            "Raspberry",
            ["raspberry"]
        ),

        (
            "Soybean",
            ["soybean"]
        ),

        (
            "Squash",
            ["squash"]
        )
    ]

    for crop, keywords in crop_order:

        for keyword in keywords:

            if keyword in normalized:

                return crop

    return "Unknown"


# =========================================================
# HEALTH / SEVERITY
# =========================================================

def calculate_health_score(
    confidence,
    predicted_label
):

    normalized = normalize_label(
        predicted_label
    )

    if "healthy" in normalized:

        return min(
            98,
            max(
                80,
                int(confidence + 8)
            )
        )

    if confidence < 50:
        return 45

    if confidence < 65:
        return 55

    if confidence < 80:
        return 65

    if confidence < 90:
        return 72

    return 82


def calculate_severity(
    confidence,
    predicted_label
):

    normalized = normalize_label(
        predicted_label
    )

    if "healthy" in normalized:

        return "Low"

    if confidence >= 90:

        return "High"

    if confidence >= 75:

        return "Medium"

    return "Low"


# =========================================================
# LOGIN GATE
# =========================================================

if not st.session_state.authenticated and not st.session_state.guest_mode:
    render_login_signup()
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title(
    "🌱 AgriCare AI"
)

if st.session_state.authenticated:
    st.sidebar.success(f"👤 Welcome, {st.session_state.display_name}")
    if st.sidebar.button("🚪 Logout", use_container_width=True, key="logout_button"):
        logout_user(); st.rerun()
elif st.session_state.guest_mode:
    st.sidebar.info("👤 Guest Mode — Offline Activity only")
    if st.sidebar.button("🔐 Login / Sign Up", use_container_width=True, key="guest_login_button"):
        st.session_state.guest_mode = False; st.session_state.quick_page = "Home"; st.rerun()

st.sidebar.caption(
    tr("smart")
)


nav_internal = [
    "Home",
    "AI Plant Doctor",
    "History",
    "New Farming",
    "Offline Activity",
    "AI Farming Chat"
]


nav_labels = [
    tr("home"),
    tr("doctor"),
    tr("history"),
    tr("new_farming"),
    tr("offline"),
    tr("chat")
]


nav_map = dict(
    zip(
        nav_labels,
        nav_internal
    )
)


# =========================================================
# QUICK ACTION NAVIGATION
# =========================================================
# Keep navigation persistent across Streamlit reruns.
# File upload/camera widgets trigger reruns; the previous one-time
# quick_page logic was resetting the app back to Home on every upload.

if st.session_state.quick_page != "Home":
    target_page = st.session_state.quick_page
    target_label = next(
        (label for label, internal in nav_map.items() if internal == target_page),
        None
    )
    if target_label is not None:
        st.session_state.navigation_choice = target_label
    st.session_state.quick_page = "Home"

selected_nav = st.sidebar.radio(
    tr("navigation"),
    nav_labels,
    key="navigation_choice"
)

page = nav_map[selected_nav]

if st.session_state.guest_mode:
    page = "Offline Activity"


st.sidebar.markdown("---")


selected_language = st.sidebar.selectbox(
    tr("language"),
    [
        "English",
        "తెలుగు",
        "हिंदी"
    ],
    index=[
        "English",
        "తెలుగు",
        "हिंदी"
    ].index(
        st.session_state.language
    ),
    key="language_selector"
)


if selected_language != st.session_state.language:

    st.session_state.language = selected_language

    st.rerun()


st.sidebar.markdown("---")


st.sidebar.info(
    "AgriCare AI helps farmers detect crop problems, "
    "plan farming activities and get AI-based guidance."
)


# =========================================================
# UNIVERSAL AI ASSISTANCE
# =========================================================
def render_universal_ai_help(section_name):
    with st.expander(f"🤖 AI Assistance — Need help with {section_name}?", expanded=False):
        st.write("Ask AgriCare AI for crop-specific guidance, explain symptoms, plan the next step, or understand the information on this page.")
        if st.button("💬 Ask AI Assistant", key=f"universal_ai_{section_name}", use_container_width=True, type="primary"):
            if st.session_state.get("guest_mode"):
                st.warning("Please log in to use AI Farming Chat. Guest mode is limited to Offline Activity.")
            else:
                st.session_state.quick_page = "AI Farming Chat"
                st.rerun()

# =========================================================
# HOME

if page == "Home":

    render_universal_ai_help("this Home page")

    # Native Streamlit components are used here instead of raw HTML.
    # This prevents HTML tags from appearing as text in the browser.

    st.title("🌱 AgriCare AI")
    st.subheader("AI-powered crop disease detection and smart farming assistant")

    st.info(tr("welcome") + "\n\n" + tr("protect"))

    st.markdown("## 📊 AgriCare AI at a Glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🌾 " + tr("supported"), "10+")

    with col2:
        st.metric("🧠 AI Classes", "38")

    with col3:
        st.metric("🌐 " + tr("languages"), "3")

    with col4:
        st.metric("🤖 " + tr("assistant"), "24/7")

    st.markdown("---")
    st.subheader(tr("main_features"))

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### " + tr("doctor_card"))
            st.write(tr("doctor_desc"))
            if st.button("Open AI Plant Doctor", key="home_doctor", use_container_width=True):
                st.session_state.quick_page = "AI Plant Doctor"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### " + tr("farm_card"))
            st.write(tr("farm_desc"))
            if st.button("Open New Farming", key="home_farming", use_container_width=True):
                st.session_state.quick_page = "New Farming"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("### " + tr("chat_card"))
            st.write(tr("chat_desc"))
            if st.button("Open AI Chat", key="home_chat", use_container_width=True):
                st.session_state.quick_page = "AI Farming Chat"
                st.rerun()

    st.markdown("## " + tr("quick_actions"))
    st.caption(tr("quick_caption"))

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🔬 " + tr("scan"))
            st.write(tr("scan_desc"))
            if st.button("🔬 Start Crop Scan", key="quick_detect", use_container_width=True, type="primary"):
                st.session_state.quick_page = "AI Plant Doctor"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown("### 🌾 " + tr("plan"))
            st.write(tr("plan_desc"))
            if st.button("🌾 Start Farm Plan", key="quick_plan", use_container_width=True, type="primary"):
                st.session_state.quick_page = "New Farming"
                st.rerun()

    with col3:
        with st.container(border=True):
            st.markdown("### 🤖 " + tr("ask"))
            st.write(tr("ask_desc"))
            if st.button("🤖 Ask AI", key="quick_chat", use_container_width=True, type="primary"):
                st.session_state.quick_page = "AI Farming Chat"
                st.rerun()

    st.markdown("---")
    st.success("🌱 AgriCare AI is ready to help you monitor crops, identify possible plant problems and plan farming activities.")


# =========================================================
# AI PLANT DOCTOR
# =========================================================

elif page == "AI Plant Doctor":

    # AI Plant Doctor already contains image-based AI assistance.
    st.title(
        tr("doctor_title")
    )

    st.write(
        tr("doctor_intro")
    )


    st.markdown("---")


    st.subheader(
        tr("crop_info")
    )


    col1, col2 = st.columns(2)


    with col1:

        crop = st.selectbox(
            tr("select_crop"),
            [
                "Rice",
                "Maize",
                "Tomato",
                "Chilli",
                "Potato",
                "Groundnut",
                "Cotton",
                "Pulses",
                "Banana",
                "Mango"
            ]
        )


        crop_age = st.number_input(
            tr("age"),
            min_value=1,
            max_value=1000,
            value=30
        )


        growth_stage = st.selectbox(
            tr("growth"),
            [
                "Seedling",
                "Vegetative",
                "Flowering",
                "Fruiting",
                "Maturity"
            ]
        )


    with col2:

        soil = st.selectbox(
            tr("soil"),
            [
                "Black Soil",
                "Red Soil",
                "Alluvial Soil",
                "Sandy Soil",
                "Clay Soil",
                "Other"
            ]
        )


        water = st.selectbox(
            tr("water"),
            [
                "Low",
                "Medium",
                "High"
            ]
        )


        season = st.selectbox(
            tr("season"),
            [
                "Kharif",
                "Rabi",
                "Summer"
            ]
        )


    st.markdown("---")


    st.subheader(
        tr("image")
    )


    # -----------------------------------------------------
    # IMAGE SOURCE
    # -----------------------------------------------------

    input_method = st.radio(
        tr("source"),
        [
            tr("upload"),
            tr("camera")
        ],
        horizontal=True
    )


    uploaded_image = None


    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    if input_method == tr("upload"):

        uploaded_image = st.file_uploader(
            tr("upload_prompt"),
            type=None,
            accept_multiple_files=False,
            key="crop_image_uploader",
            help="Choose a JPG, JPEG, PNG, WEBP or BMP image. The app will validate the actual image format."
        )


    # -----------------------------------------------------
    # CAMERA
    # -----------------------------------------------------

    else:

        uploaded_image = st.camera_input(
            tr("camera_prompt"),
            key="crop_camera"
        )


    # -----------------------------------------------------
    # PREPARE IMAGE
    # -----------------------------------------------------

    current_image = None

    if uploaded_image is not None:

        try:

            current_image = read_uploaded_image(
                uploaded_image
            )

            valid, message = image_is_reasonable(
                current_image
            )

            if not valid:

                st.warning(
                    "⚠️ " + message
                )

            else:

                # Save bytes in session so the image
                # remains available after reruns.
                st.session_state.last_image_bytes = (
                    uploaded_image.getvalue()
                )

                st.image(
                    current_image,
                    caption=f"{tr('selected')} — {current_image.info.get('detected_format', 'Image')}",
                    use_container_width=True
                )

        except Exception as e:

            current_image = None

            st.error(
                f"❌ Image upload error: {e}"
            )


    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    st.markdown("---")


    analyze = st.button(
        tr("analyze"),
        use_container_width=True,
        type="primary",
        key="analyze_crop"
    )


    # =====================================================
    # AI ANALYSIS
    # =====================================================

    if analyze:

        if current_image is None:

            st.warning(
                tr("need_image")
            )

        else:

            try:

                valid, message = image_is_reasonable(
                    current_image
                )

                if not valid:

                    st.warning(
                        "⚠️ " + message
                    )

                else:

                    with st.spinner(
                        tr("analyzing")
                    ):

                        prediction = run_prediction(
                            current_image,
                            selected_crop=crop
                        )

                    predicted_label = prediction["label"]
                    confidence = prediction["confidence"]

                    supported_by_model = ["Maize", "Tomato", "Chilli", "Potato"]
                    model_supported_for_selected_crop = crop in supported_by_model

                    if model_supported_for_selected_crop and prediction.get("crop_gated"):
                        # The farmer's selected crop is authoritative for the crop-specific
                        # disease classifier. Do not compare it with an unrelated raw top class.
                        ai_crop = crop
                        crop_match = "Matched — crop-specific AI analysis"
                    else:
                        ai_crop = prediction.get("raw_crop", "Unknown")
                        crop_match = "Not reliably supported by current AI model"


                    health_score = (
                        calculate_health_score(
                            confidence,
                            predicted_label
                        )
                    )


                    severity = (
                        calculate_severity(
                            confidence,
                            predicted_label
                        )
                    )


                    guidance = get_crop_disease_guidance(
                        crop,
                        predicted_label
                    )


                    disease = guidance["name"]




                    # -------------------------------------------------
                    # SAVE SESSION DATA
                    # -------------------------------------------------

                    st.session_state.analysis_done = True

                    st.session_state.scan_saved = False

                    st.session_state.voice_audio_path = None


                    st.session_state.analysis_data = {

                        "crop": crop,

                        "ai_crop": ai_crop,

                        "crop_match": crop_match,

                        "crop_age": crop_age,

                        "growth_stage": growth_stage,

                        "soil": soil,

                        "water": water,

                        "season": season,

                        "health_score": health_score,

                        "confidence": confidence,

                        "reliability": prediction.get("reliability", confidence),

                        "consistency": prediction.get("consistency", 100),

                        "image_quality": prediction.get("image_quality", {}),

                        "disease": disease,

                        "predicted_label": predicted_label,

                        "severity": severity,

                        "problem_detail": guidance.get("problem", ""),

                        "medicine_type": guidance.get("medicine_type", ""),

                        "active_ingredients": guidance.get("active_ingredients", []),

                        "management": " ".join(
                            guidance.get("treatment", guidance.get("management", []))
                        ),

                        "prevention": " ".join(
                            guidance.get("prevention", [])
                        ),

                        "top_predictions":
                            prediction[
                                "top_predictions"
                            ]
                    }


                    # -------------------------------------------------
                    # SAVE HISTORY
                    # -------------------------------------------------

                    save_scan(
                        st.session_state.analysis_data
                    )

                    st.session_state.scan_saved = True


            except Exception as e:

                st.session_state.analysis_done = False

                st.error(
                    f"{tr('model_error')} {e}"
                )


    # =====================================================
    # SHOW RESULT
    # =====================================================

    if st.session_state.analysis_done:

        data = (
            st.session_state.analysis_data
        )


        crop = data["crop"]

        ai_crop = data["ai_crop"]

        crop_match = data["crop_match"]

        crop_age = data["crop_age"]

        growth_stage = data["growth_stage"]

        soil = data["soil"]

        water = data["water"]

        season = data["season"]

        health_score = data["health_score"]

        confidence = data["confidence"]

        reliability = data.get("reliability", confidence)

        consistency = data.get("consistency", 100)

        image_quality = data.get("image_quality", {})

        disease = data["disease"]

        predicted_label = data["predicted_label"]

        severity = data["severity"]

        management = data["management"]

        prevention = data["prevention"]

        problem_detail = data.get("problem_detail", "")

        top_predictions = data[
            "top_predictions"
        ]


        st.success(
            tr("analyzed")
        )


        if st.session_state.get(
            "scan_saved"
        ):

            st.info(
                tr("saved")
            )


        # -------------------------------------------------
        # RELIABILITY WARNING
        # -------------------------------------------------

        if reliability < 60:
            st.warning(
                "⚠️ **Low detection reliability.** The model is not stable enough to treat this result as a diagnosis. "
                "Retake a sharp close-up photo showing the whole leaf and affected area."
            )
        elif reliability < 75:
            st.info(
                "ℹ️ **Moderate detection reliability.** Compare the visible symptoms with the result and verify before chemical treatment."
            )
        else:
            st.success(
                "✅ **Good screening reliability.** This is still an AI-assisted screening result, not a guaranteed field diagnosis."
            )

        qtext = image_quality.get("quality", "Unknown") if isinstance(image_quality, dict) else "Unknown"
        st.caption(f"📷 Image quality: {qtext}  •  🔄 Original/flip consistency: {consistency}%  •  🧠 Classifier confidence: {confidence}%")

        if consistency < 70:
            st.warning("🔄 The AI gave noticeably different results after a small image change. Treat this scan as uncertain and retake the photo.")


        # -------------------------------------------------
        # RESULT
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            tr("result")
        )

        # Hackathon-friendly visual confidence indicator
        st.markdown("### 🩺 Plant Health Summary")
        if health_score >= 80:
            health_message = "🌿 Plant appears relatively healthy."
        elif health_score >= 60:
            health_message = "🟡 Possible plant stress or disease detected."
        else:
            health_message = "🔴 Possible significant plant health problem detected."
        st.write(health_message)
        st.progress(min(max(health_score, 0), 100) / 100)

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                tr("health"),
                f"{health_score}/100"
            )


        with col2:

            st.metric(
                tr("confidence"),
                f"{confidence}%"
            )


        with col3:

            st.metric(
                tr("disease"),
                disease
            )


        with col4:

            st.metric(
                tr("severity"),
                severity
            )


        # -------------------------------------------------
        # AI RESULT CARD
        # -------------------------------------------------
        # Use native Streamlit components here instead of raw HTML.
        # This prevents HTML tags from appearing as visible text.
        st.markdown("### 🧠 AI Model Details")

        with st.container(border=True):
            st.caption("The model provides an AI-assisted screening result. It should be verified in real field conditions.")
            st.write("**AI MODEL PREDICTION**")
            st.markdown(f"### {predicted_label}")

            st.write("**AI CROP IDENTIFICATION**")
            st.markdown(f"### {ai_crop}")

            st.write("**CONFIDENCE**")
            st.markdown(f"### {confidence}%")
            st.write("**DETECTION RELIABILITY")
            st.markdown(f"### {reliability}%")

        # -------------------------------------------------
        # CROP MISMATCH
        # -------------------------------------------------

        if ai_crop != "Unknown":

            if (
                ai_crop.lower()
                == crop.lower()
            ):

                st.success(
                    f"🌱 Crop check: Selected crop "
                    f"**{crop}** matches the AI detected "
                    f"crop **{ai_crop}**."
                )

            else:

                st.warning(
                    f"⚠️ Crop mismatch detected. "
                    f"You selected **{crop}**, but the "
                    f"image model suggests **{ai_crop}**."
                )

        else:

            st.info(
                "ℹ️ The model could not confidently "
                "identify the crop species from this image."
            )


        # -------------------------------------------------
        # UNSUPPORTED CROP WARNING
        # -------------------------------------------------

        supported_by_model = [
            "Maize",
            "Tomato",
            "Chilli",
            "Potato"
        ]


        if crop not in supported_by_model:

            st.warning(
                f"⚠️ **{crop} is not a dedicated class in the current image model.** "
                "AgriCare AI will NOT claim that a disease from another crop is your disease. "
                "For this crop, use the AI Farming Chat with symptoms and a clear image, "
                "and confirm any chemical treatment with a local agricultural expert."
            )
        else:
            st.success(
                f"✅ Crop-specific AI mode active for **{crop}**. "
                "Only disease/healthy classes belonging to this selected crop are compared, "
                "so unrelated crop classes are excluded from the final result."
            )


        # -------------------------------------------------
        # DETAILS
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            tr("details")
        )


        st.write(
            f"**{tr('selected_crop')}** {crop}"
        )

        st.write(
            f"**{tr('ai_crop')}** {ai_crop}"
        )

        st.write(
            f"**{tr('crop_match')}** {crop_match}"
        )

        st.write(
            f"**{tr('crop_age')}** {crop_age} days"
        )

        st.write(
            f"**{tr('growth_stage')}** {growth_stage}"
        )

        st.write(
            f"**{tr('soil')}** {soil}"
        )

        st.write(
            f"**{tr('water_availability')}** {water}"
        )

        st.write(
            f"**{tr('season')}** {season}"
        )

        st.write(
            f"**{tr('problem')}** {predicted_label}"
        )

        st.write(
            f"**{tr('disease_conf')}** {confidence}%"
        )

        st.write(
            f"**{tr('severity')}** {severity}"
        )


        # -------------------------------------------------
        # TOP 3
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🔎 Top AI Predictions"
        )


        for index, item in enumerate(
            top_predictions,
            start=1
        ):

            st.write(
                f"**{index}. {item['label']}** — "
                f"{item['confidence']}%"
            )


        # -------------------------------------------------
        # MANAGEMENT
        # -------------------------------------------------

        st.markdown("---")
        st.markdown("### 🚑 Recommended Next Steps")
        st.write("1. Take a clear close-up photo of the affected leaf.")
        st.write("2. Compare the AI result with visible symptoms in the field.")
        st.write("3. Follow the management suggestions below.")
        st.write("4. If symptoms spread quickly, consult a local agriculture expert.")

        st.subheader(
            tr("management")
        )

        if problem_detail:
            st.info("🧾 **What this means:** " + problem_detail)

        for item in guidance.get("treatment", guidance.get("management", [])):
            st.write("• " + item)

        # -------------------------------------------------
        # MEDICINE DATABASE
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("💊 Recommended Medicine / Active Ingredients")

        medicine_type = guidance.get("medicine_type", "Crop-specific management")
        active_ingredients = guidance.get("active_ingredients", [])

        st.info(f"**Medicine type:** {medicine_type}")

        if active_ingredients:
            st.write("**Suggested active ingredients / options:**")
            for ingredient in active_ingredients:
                st.write("• " + ingredient)
        else:
            st.write("• No disease-specific medicine is recommended from this scan.")

        st.warning(
            "⚠️ Medicine safety: Active ingredients shown here are guidance, not a prescription. "
            "Use only a product currently registered for the selected crop and confirmed problem. "
            "Follow the product label for concentration, dose, spray interval, PPE and pre-harvest interval. "
            "Do not mix products unless the label permits it."
        )


        # -------------------------------------------------
        # PREVENTION
        # -------------------------------------------------

        st.markdown(
            "### " + tr("prevention")
        )


        for item in guidance.get("prevention", []):
            st.write("• " + item)


        # -------------------------------------------------
        # CONTEXT
        # -------------------------------------------------

        st.markdown("---")

        st.info(
            "🌾 Context used: crop age, growth stage, "
            "soil, water availability and season are shown "
            "to support the recommendation layer. "
            "The disease prediction itself comes from "
            "the uploaded image."
        )


        st.warning(
            "⚠️ This is an AI-assisted screening tool, "
            "not a laboratory diagnosis. Image quality and "
            "real field conditions can affect accuracy."
        )


        # -------------------------------------------------
        # VOICE
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            tr("voice")
        )


        voice_language = st.selectbox(
            tr("voice_lang"),
            [
                "English",
                "తెలుగు",
                "हिंदी"
            ],
            key="voice_language"
        )


        language_map = {

            "English": "en",

            "తెలుగు": "te",

            "हिंदी": "hi"

        }


        if st.button(
            tr("generate"),
            use_container_width=True,
            key="generate_voice"
        ):


            if voice_language == "English":

                result_text = f"""
                Crop selected: {crop}.
                AI detected crop: {ai_crop}.
                The AI prediction is {predicted_label}.
                Confidence is {confidence} percent.
                Plant health score is {health_score} out of 100.
                Severity is {severity}.
                Recommended management:
                {management}.
                Prevention:
                {prevention}.
                """


            elif voice_language == "తెలుగు":

                result_text = f"""
                ఎంచుకున్న పంట {crop}.
                AI గుర్తించిన పంట {ai_crop}.
                AI గుర్తించిన సమస్య {predicted_label}.
                నమ్మక స్థాయి {confidence} శాతం.
                మొక్క ఆరోగ్య స్కోర్ 100కి {health_score}.
                తీవ్రత {severity}.
                సూచించిన నిర్వహణ:
                {management}.
                నివారణ:
                {prevention}.
                """


            else:

                result_text = f"""
                चुनी गई फसल {crop} है।
                AI द्वारा पहचानी गई फसल {ai_crop} है।
                AI द्वारा पहचानी गई समस्या {predicted_label} है।
                विश्वास स्तर {confidence} प्रतिशत है।
                पौधे का स्वास्थ्य स्कोर 100 में से {health_score} है।
                गंभीरता {severity} है।
                सुझाया गया प्रबंधन:
                {management}.
                रोकथाम:
                {prevention}.
                """


            try:

                lang_code = language_map[
                    voice_language
                ]


                with st.spinner(
                    tr("creating")
                ):

                    tts = gTTS(
                        text=result_text,
                        lang=lang_code,
                        slow=False
                    )


                    audio_file = (
                        tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=".mp3"
                        )
                    )


                    tts.save(
                        audio_file.name
                    )


                    audio_file.close()


                st.session_state.voice_audio_path = (
                    audio_file.name
                )


                st.session_state.voice_language_used = (
                    voice_language
                )


            except Exception as e:

                st.session_state.voice_audio_path = None

                st.error(
                    f"❌ Voice generation failed: {e}"
                )


        # -------------------------------------------------
        # AUDIO PLAYER
        # -------------------------------------------------

        if st.session_state.get(
            "voice_audio_path"
        ):

            st.success(
                tr("voice_ok")
            )

            st.write(
                tr("audio")
            )

            st.audio(
                st.session_state.voice_audio_path,
                format="audio/mp3"
            )

            st.info(
                tr("play")
            )


        # -------------------------------------------------
        # FEEDBACK
        # -------------------------------------------------

        st.markdown("---")

        st.subheader(
            tr("feedback")
        )


        feedback = st.radio(
            tr("useful"),
            [
                "Yes 👍",
                "Partially 😐",
                "No 👎"
            ],
            horizontal=True
        )


        rating = st.slider(
            tr("rating"),
            min_value=1,
            max_value=5,
            value=4
        )


        comment = st.text_area(
            tr("comment")
        )


        if st.button(
            tr("submit"),
            use_container_width=True,
            key="submit_feedback"
        ):

            st.success(
                tr("thanks")
            )


# =========================================================
# HISTORY
# =========================================================

elif page == "History":

    render_universal_ai_help("your farming history")

    st.title(
        tr("history_title")
    )

    st.write(
        tr("history_intro")
    )


    tab1, tab2, tab3 = st.tabs(
        [
            "🔬 Scans",
            "🌾 Farm Plans",
            "💬 Questions"
        ]
    )


    with tab1:

        st.subheader(
            "🔬 Previous Crop Scans"
        )


        scans = get_scans()


        if not scans:

            st.info(
                "No previous scans saved yet. "
                "Analyze a crop in AI Plant Doctor "
                "and it will automatically appear here."
            )


        else:

            st.success(
                f"📊 {len(scans)} scan(s) saved"
            )


            for row in scans:

                (
                    scan_id,
                    scan_date,
                    h_crop,
                    h_age,
                    h_stage,
                    h_soil,
                    h_water,
                    h_season,
                    h_health,
                    h_confidence,
                    h_disease,
                    h_severity,
                    h_management
                ) = row


                with st.expander(
                    f"🌱 {h_crop} | "
                    f"🦠 {h_disease} | "
                    f"⚠️ {h_severity} | "
                    f"📅 {scan_date}"
                ):


                    col1, col2, col3, col4 = st.columns(4)


                    with col1:

                        st.metric(
                            "Health",
                            f"{h_health}/100"
                        )


                    with col2:

                        st.metric(
                            "Confidence",
                            f"{h_confidence}%"
                        )


                    with col3:

                        st.metric(
                            "Disease",
                            h_disease
                        )


                    with col4:

                        st.metric(
                            "Severity",
                            h_severity
                        )


                    st.markdown(
                        "### 🌾 Crop Details"
                    )


                    st.write(
                        f"**Crop:** {h_crop}"
                    )

                    st.write(
                        f"**Crop Age:** {h_age} days"
                    )

                    st.write(
                        f"**Growth Stage:** {h_stage}"
                    )

                    st.write(
                        f"**Soil:** {h_soil}"
                    )

                    st.write(
                        f"**Water:** {h_water}"
                    )

                    st.write(
                        f"**Season:** {h_season}"
                    )


                    st.markdown(
                        "### 🩺 Recommended Management"
                    )


                    st.write(
                        h_management
                    )


            st.markdown("---")


            st.caption(
                "💡 History is stored locally using SQLite "
                "on this application."
            )


    with tab2:

        st.subheader(
            "🌾 Farm Plans"
        )

        st.info(
            "Farm-plan history will be added in the next step."
        )


    with tab3:

        st.subheader(
            "💬 AI Questions"
        )

        st.info(
            "AI farming-question history will be added "
            "in a later step."
        )


# =========================================================
# NEW FARMING
# =========================================================

elif page == "New Farming":

    render_universal_ai_help("your new farming plan")

    st.title(
        tr("new_title")
    )

    st.write(
        "Create a personalized farming plan."
    )

    st.markdown("---")


    st.subheader(
        tr("farm_info")
    )


    col1, col2 = st.columns(2)


    with col1:

        farm_crop = st.selectbox(
            tr("select_crop"),
            [
                "Rice",
                "Maize",
                "Tomato",
                "Chilli",
                "Potato",
                "Groundnut",
                "Cotton",
                "Pulses",
                "Banana",
                "Mango"
            ],
            key="farm_crop"
        )


        start_date = st.date_input(
            "Farming Start Date",
            value=date.today()
        )


        farm_size = st.number_input(
            "Farm Size (acres)",
            min_value=0.1,
            value=1.0,
            step=0.1
        )


        farm_soil = st.selectbox(
            tr("soil"),
            [
                "Black Soil",
                "Red Soil",
                "Alluvial Soil",
                "Sandy Soil",
                "Clay Soil"
            ],
            key="farm_soil"
        )


    with col2:

        irrigation = st.selectbox(
            "Irrigation Type",
            [
                "Rainfed",
                "Drip",
                "Sprinkler",
                "Flood Irrigation"
            ]
        )


        farm_water = st.selectbox(
            tr("water"),
            [
                "Low",
                "Medium",
                "High"
            ],
            key="farm_water"
        )


        farming_method = st.selectbox(
            "Farming Method",
            [
                "Traditional",
                "Organic",
                "Modern",
                "Mixed"
            ]
        )


        previous_crop = st.text_input(
            "Previous Crop"
        )


    budget = st.number_input(
        "Approximate Budget (₹)",
        min_value=0,
        value=10000,
        step=1000
    )


    st.markdown("---")


    if st.button(
        "🚜 Generate Farming Plan",
        use_container_width=True,
        type="primary"
    ):


        st.success(
            "✅ Farming plan generated!"
        )


        st.subheader(
            f"🌱 {farm_crop} Farming Plan"
        )


        st.write(
            f"**Start Date:** {start_date}"
        )

        st.write(
            f"**Farm Size:** {farm_size} acres"
        )

        st.write(
            f"**Soil:** {farm_soil}"
        )

        st.write(
            f"**Irrigation:** {irrigation}"
        )

        st.write(
            f"**Water:** {farm_water}"
        )

        st.write(
            f"**Farming Method:** {farming_method}"
        )

        st.write(
            f"**Previous Crop:** {previous_crop}"
        )

        st.write(
            f"**Budget:** ₹{budget}"
        )


        st.markdown("---")


        st.subheader(
            "📅 Farming Steps"
        )


        st.markdown("""
        **Phase 1 — Land Preparation**

        Prepare the field and remove weeds.

        **Phase 2 — Seed/Planting**

        Use healthy and suitable planting material.

        **Phase 3 — Crop Growth**

        Monitor water, weeds, pests and diseases.

        **Phase 4 — Flowering/Fruiting**

        Closely monitor crop health and nutrition.

        **Phase 5 — Harvest**

        Harvest at the recommended maturity stage.
        """)


        st.info(
            "💡 Future version will generate a detailed plan "
            "using local soil, weather and crop information."
        )


# =========================================================
# OFFLINE ACTIVITY - CROP SPECIFIC REAL IMAGES
# =========================================================

elif page == "Offline Activity":

    render_universal_ai_help("the offline learning guide")

    st.title(tr("offline_title"))

    st.write(
        "📚 Crop-specific field guide with real disease/pest reference photos. "
        "Images are downloaded once and then cached locally for offline use."
    )

    st.markdown("---")

    from pathlib import Path
    from urllib.request import Request, urlopen

    offline_assets = Path("offline_assets")
    offline_assets.mkdir(exist_ok=True)

    # Real reference photographs from agricultural / educational sources.
    # The first successful load downloads them into offline_assets.
    crop_library = {
        "Rice": {
            "problem": "Bacterial Leaf Blight",
            "cause": "Bacterium — Xanthomonas oryzae pv. oryzae",
            "symptoms": [
                "Water-soaked or yellowish stripes starting near leaf tips or margins.",
                "Lesions extend along the leaf and may become straw-white/brown.",
                "Severe plants can wilt, dry and become stunted."
            ],
            "management": [
                "Use healthy or certified seed and resistant varieties where available.",
                "Avoid excessive nitrogen application.",
                "Keep irrigation water from moving from heavily infected areas to healthy areas.",
                "Scout the field regularly, especially after rainy/windy weather."
            ],
            "prevention": [
                "Good field sanitation",
                "Balanced nitrogen management",
                "Disease-free seed",
                "Regular field scouting"
            ],
            "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Bacterial_blight_of_rice.jpeg",
            "source": "Wikimedia Commons / Bugwood",
            "credit": "Bacterial blight of rice"
        },
        "Maize": {
            "problem": "Fall Armyworm",
            "cause": "Insect pest — Spodoptera frugiperda",
            "symptoms": [
                "Shot holes and ragged feeding damage on young leaves.",
                "Frass may be visible inside the whorl.",
                "Larvae can damage the growing point and developing cobs."
            ],
            "management": [
                "Inspect whorls and young leaves frequently.",
                "Use integrated pest management and conserve natural enemies.",
                "Remove heavily infested plant material when practical.",
                "Use only locally recommended registered products when treatment is necessary."
            ],
            "prevention": [
                "Early scouting",
                "Field sanitation",
                "IPM and natural enemies",
                "Avoid unnecessary repeated insecticide use"
            ],
            "image": "https://rbmagazine-assets.s3.ap-southeast-2.amazonaws.com/media/news/image/Cesar_Fall_armyworm_Credit_Melina_Miles_QDAF.jpg",
            "source": "Rural Business Media / QDAF",
            "credit": "Fall armyworm on maize"
        },
        "Tomato": {
            "problem": "Late Blight",
            "cause": "Oomycete — Phytophthora infestans",
            "symptoms": [
                "Dark brown to black irregular lesions on leaves.",
                "Lesions can expand quickly under cool, wet conditions.",
                "Fruit and stems can also develop dark, damaged areas."
            ],
            "management": [
                "Remove badly affected leaves and plant debris.",
                "Improve airflow and avoid prolonged leaf wetness.",
                "Scout frequently during wet weather.",
                "Use locally recommended disease-management products only when needed."
            ],
            "prevention": [
                "Good airflow",
                "Avoid overhead irrigation where possible",
                "Remove infected debris",
                "Frequent scouting after rain"
            ],
            "image": "https://live.staticflickr.com/5115/5816170621_4fd6376d8e_o.jpg",
            "source": "Wikimedia Commons / Scot Nelson",
            "credit": "Tomato late blight"
        },
        "Chilli": {
            "problem": "Chilli Leaf Curl Virus",
            "cause": "Virus group — begomoviruses; commonly spread by whiteflies",
            "symptoms": [
                "Upward curling of leaf margins.",
                "Yellowing veins and reduced leaf size.",
                "Shortened internodes and stunted growth.",
                "Fruit formation may be poor or distorted."
            ],
            "management": [
                "Monitor and reduce whitefly populations using integrated pest management.",
                "Remove severely infected plants early where appropriate.",
                "Control weeds and alternate hosts around the field.",
                "Use yellow sticky traps for monitoring whiteflies."
            ],
            "prevention": [
                "Healthy planting material",
                "Whitefly monitoring",
                "Weed control",
                "Regular early-season scouting"
            ],
            "image": "https://content.peat-cloud.com/w400/chilli-leaf-curl-virus-pepper-1574948117.jpg",
            "source": "Plantix",
            "credit": "Chilli leaf curl virus"
        },
        "Potato": {
            "problem": "Late Blight",
            "cause": "Oomycete — Phytophthora infestans",
            "symptoms": [
                "Large brown or dark lesions on leaves.",
                "Yellow-green margins can surround early lesions.",
                "Disease can spread rapidly in humid/wet conditions."
            ],
            "management": [
                "Scout leaves frequently during wet weather.",
                "Remove or manage infected plant material according to local guidance.",
                "Maintain good field hygiene.",
                "Use locally recommended registered disease-control products when required."
            ],
            "prevention": [
                "Healthy seed tubers",
                "Field scouting",
                "Good drainage",
                "Avoid prolonged leaf wetness"
            ],
            "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Potato_Late_Blight.JPG",
            "source": "Wikimedia Commons / SLU",
            "credit": "Potato late blight"
        },
        "Groundnut": {
            "problem": "Groundnut Leaf Spot",
            "cause": "Commonly fungal leaf-spot diseases such as Cercospora spp.",
            "symptoms": [
                "Brown circular or irregular spots on leaflets.",
                "Yellow halos can develop around lesions.",
                "Severe infection may cause premature leaf loss."
            ],
            "management": [
                "Scout lower and middle canopy leaves regularly.",
                "Use crop rotation and field sanitation.",
                "Avoid unnecessary prolonged leaf wetness.",
                "Follow locally recommended fungicide programs if disease pressure is high."
            ],
            "prevention": [
                "Crop rotation",
                "Healthy seed",
                "Field sanitation",
                "Early scouting"
            ],
            "image": "https://www.omafra.gov.on.ca/CropOp/images/crop_images/specialty-fruit/nuts/pean/peanf9_zoom.jpg",
            "source": "Ontario Ministry of Agriculture",
            "credit": "Peanut leaf spot"
        },
        "Cotton": {
            "problem": "Cotton Bollworm",
            "cause": "Insect pest — Helicoverpa armigera",
            "symptoms": [
                "Caterpillars feed on squares, flowers and bolls.",
                "Boll entry holes and frass may be visible.",
                "Damaged bolls can lose lint and seed quality."
            ],
            "management": [
                "Scout squares, flowers and bolls regularly.",
                "Use pheromone/trap monitoring where available.",
                "Protect beneficial insects and use IPM.",
                "Use only locally recommended registered products when thresholds justify treatment."
            ],
            "prevention": [
                "Regular scouting",
                "Trap monitoring",
                "IPM",
                "Avoid unnecessary broad-spectrum sprays"
            ],
            "image": "https://eu-images.contentstack.com/v3/assets/blte5a51c2d28bbcc9c/blt0c60162614b6445e/64878b402b13a6c83191c792/CSIRO_20Helicoverpa-armigera.jpg",
            "source": "CSIRO / Contentstack image",
            "credit": "Cotton bollworm in boll"
        },
        "Pulses": {
            "problem": "Pigeonpea Fusarium Wilt",
            "cause": "Soil-borne fungus — Fusarium udum",
            "symptoms": [
                "Leaves become pale or yellow before wilting.",
                "Branches and whole plants may dry progressively.",
                "Affected plants may appear in patches within the field."
            ],
            "management": [
                "Remove severely affected plants where practical.",
                "Use resistant/tolerant varieties where available.",
                "Use healthy seed and suitable seed-treatment practices.",
                "Rotate crops to reduce repeated disease pressure."
            ],
            "prevention": [
                "Resistant varieties",
                "Healthy seed",
                "Crop rotation",
                "Field sanitation"
            ],
            "image": "https://content.peat-cloud.com/w400/fusarium-wilt-pigeonpea-1681484960.jpg",
            "source": "Plantix",
            "credit": "Fusarium wilt of pigeonpea"
        },
        "Banana": {
            "problem": "Black Sigatoka",
            "cause": "Fungus — Pseudocercospora fijiensis",
            "symptoms": [
                "Small dark streaks or lesions appear on banana leaves.",
                "Lesions enlarge and may merge into large necrotic areas.",
                "Severe infection can reduce green leaf area."
            ],
            "management": [
                "Remove badly affected leaves according to local recommendations.",
                "Maintain good plantation sanitation and airflow.",
                "Monitor new leaves frequently.",
                "Follow locally recommended disease-management programs."
            ],
            "prevention": [
                "Sanitation",
                "Good airflow",
                "Regular leaf monitoring",
                "Healthy planting material"
            ],
            "image": "https://commons.wikimedia.org/wiki/Special:FilePath/Banana-_Black_leaf_streak_(Black_sigatoka)_-_26038332165.jpg",
            "source": "Wikimedia Commons",
            "credit": "Banana black leaf streak / Black Sigatoka"
        },
        "Mango": {
            "problem": "Powdery Mildew",
            "cause": "Fungal disease — powdery mildew fungi",
            "symptoms": [
                "White powdery growth appears on young leaves, flowers or panicles.",
                "Affected tissue may become distorted or dry.",
                "Flower infection can reduce fruit set."
            ],
            "management": [
                "Improve canopy airflow through suitable pruning.",
                "Monitor flowers and young growth closely.",
                "Remove badly affected material where practical.",
                "Use locally recommended fungicide programs when necessary."
            ],
            "prevention": [
                "Canopy ventilation",
                "Regular scouting",
                "Balanced nutrition",
                "Timely disease management"
            ],
            "image": "https://us-central1-plantix-8e0ce.cloudfunctions.net/v1/image/w400/93f83069-792c-41ac-b550-89910cba6c3a",
            "source": "Plantix",
            "credit": "Powdery mildew on mango leaf"
        },
        "Soybean": {
            "problem": "Asian Soybean Rust",
            "cause": "Fungus — Phakopsora pachyrhizi",
            "symptoms": [
                "Small tan, reddish-brown or rust-colored spots on leaves.",
                "Pustules may be visible on the underside of leaves with a hand lens.",
                "Severe infection can cause yellowing and premature defoliation."
            ],
            "management": [
                "Inspect lower canopy leaves during flowering and pod development.",
                "Use a hand lens to check suspicious lesions.",
                "Use locally recommended fungicide programs when disease risk is confirmed.",
                "Do not rely on leaf color alone because several diseases look similar."
            ],
            "prevention": [
                "Regular scouting",
                "Weather/risk monitoring",
                "Early confirmation",
                "Use resistant/tolerant varieties where available"
            ],
            "image": "https://indexiscdn.nyc3.digitaloceanspaces.com/sites/sucessonocampo/2024/09/25024619/detalhe-ferrugem-scaled-1.webp",
            "source": "Sucesso no Campo",
            "credit": "Soybean rust"
        },
        "Grape": {
            "problem": "Downy Mildew",
            "cause": "Oomycete — Plasmopara viticola",
            "symptoms": [
                "Yellowish oil-spot lesions appear on upper leaf surfaces.",
                "White downy growth may appear on the underside under humid conditions.",
                "Severe infection can cause browning and leaf loss."
            ],
            "management": [
                "Maintain canopy airflow.",
                "Scout after rainfall and periods of high humidity.",
                "Remove heavily affected tissue where appropriate.",
                "Use locally recommended disease-management products when needed."
            ],
            "prevention": [
                "Canopy ventilation",
                "Good drainage",
                "Frequent scouting",
                "Timely management after wet weather"
            ],
            "image": "https://www.ivr.si/app/uploads/2022/12/peronospora_naslovna-scaled.jpg",
            "source": "IVR Slovenia",
            "credit": "Grapevine downy mildew"
        }
    }

    crop_names = list(crop_library.keys())

    offline_crop = st.selectbox(
        tr("select_crop"),
        crop_names,
        key="offline_crop"
    )

    data = crop_library[offline_crop]

    st.subheader(f"🌱 {offline_crop} Knowledge Library")

    # Download/cache the real image once.
    safe_name = (
        offline_crop.lower()
        .replace(" ", "_")
        .replace("/", "_")
    )
    image_path = offline_assets / f"{safe_name}_reference.jpg"

    def cache_real_image(url, destination):
        if destination.exists() and destination.stat().st_size > 5000:
            return True

        try:
            request = Request(
                url,
                headers={"User-Agent": "AgriCare-AI/1.0"}
            )
            with urlopen(request, timeout=15) as response:
                image_bytes = response.read()

            if len(image_bytes) < 5000:
                return False

            destination.write_bytes(image_bytes)
            return True
        except Exception:
            return False

    image_ready = cache_real_image(
        data["image"],
        image_path
    )

    if image_ready:
        st.image(
            str(image_path),
            caption=f"Real reference image: {data['credit']}",
            use_container_width=True
        )
        st.caption(
            f"Image source: {data['source']} | "
            "Reference image for education only."
        )
    else:
        st.warning(
            "The real reference image could not be downloaded right now. "
            "The crop-specific information below is still available. "
            "Reconnect to the internet once to cache the image for offline use."
        )

    st.markdown("---")

    left, right = st.columns(2)

    with left:
        st.markdown("### 🦠 Main Problem")
        st.info(data["problem"])

        st.markdown("### 🔬 What to Look For")
        for item in data["symptoms"]:
            st.markdown(f"- {item}")

    with right:
        st.markdown("### 🛠️ Management")
        for item in data["management"]:
            st.markdown(f"- {item}")

        st.markdown("### 🛡️ Prevention")
        for item in data["prevention"]:
            st.markdown(f"- {item}")

    st.markdown("---")

    st.subheader("📸 Early Detection Checklist")

    check1, check2, check3 = st.columns(3)

    with check1:
        st.success("1️⃣ Check new leaves")
        st.caption("Look for unusual spots, curling, yellowing or distortion.")

    with check2:
        st.success("2️⃣ Check leaf underside")
        st.caption("Pests and disease signs can be easier to see underneath leaves.")

    with check3:
        st.success("3️⃣ Compare with a healthy plant")
        st.caption("Compare symptoms across nearby plants before deciding on treatment.")

    st.warning(
        "⚠️ Important: A reference photograph is not a diagnosis. "
        "Many diseases and nutrient problems can look similar. "
        "Use AI Plant Doctor with a clear image and confirm serious cases "
        "with a local agriculture expert before applying chemicals."
    )

    st.success(
        f"📴 {offline_crop} guide is now crop-specific. "
        "After the reference image is cached once, it can be viewed without internet."
    )


# =========================================================
# AI FARMING CHAT
# =========================================================

elif page == "AI Farming Chat":

    st.title("💬 AI Farming Assistant")

    st.write(
        "Get practical, crop-aware guidance about diseases, pests, watering, soil, "
        "nutrition and plant health."
    )

    st.info(
        "🤖 Smart Assistant: This hackathon demo uses a built-in agricultural knowledge base "
        "and your selected crop/context. For visual disease screening, use AI Plant Doctor."
    )

    # ---------------------------------------------------------
    # CONTEXT PANEL
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("🌾 Tell the assistant about your crop")

    chat_language = st.selectbox(
        "🌐 AI Chat Language",
        ["English", "తెలుగు", "हिंदी"],
        key="chat_language"
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        chat_crop = st.selectbox(
            "Crop",
            ["Not specified", "Rice", "Maize", "Tomato", "Chilli", "Potato",
             "Groundnut", "Cotton", "Pulses", "Banana", "Mango", "Soybean", "Grape"],
            key="chat_crop"
        )
    with c2:
        chat_stage = st.selectbox(
            "Growth stage",
            ["Not specified", "Seedling", "Vegetative", "Flowering", "Fruiting", "Maturity"],
            key="chat_stage"
        )
    with c3:
        chat_duration = st.selectbox(
            "Problem duration",
            ["Not specified", "Less than 2 days", "2–7 days", "1–3 weeks", "More than 3 weeks"],
            key="chat_duration"
        )

    # ---------------------------------------------------------
    # QUICK QUESTIONS
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("⚡ Quick Questions")
    q1, q2, q3, q4 = st.columns(4)
    quick_question = None

    with q1:
        if st.button("🌱 Yellow leaves", use_container_width=True):
            quick_question = "Why are my crop leaves turning yellow?"
    with q2:
        if st.button("🐛 Pest problem", use_container_width=True):
            quick_question = "How can I control pests in my crop?"
    with q3:
        if st.button("💧 Watering", use_container_width=True):
            quick_question = "How should I manage watering?"
    with q4:
        if st.button("🦠 Disease", use_container_width=True):
            quick_question = "What should I do if I suspect a crop disease?"

    user_question = st.text_area(
        "🌾 Ask your farming question",
        value=quick_question or "",
        placeholder="Example: My chilli leaves are curling and I can see small insects underneath. What should I check?",
        height=130,
        key="farming_chat_question"
    )

    # ---------------------------------------------------------
    # CROP-SPECIFIC KNOWLEDGE
    # ---------------------------------------------------------
    crop_guidance = {
        "rice": {
            "disease": "For rice, inspect leaves for water-soaked or brown lesions and check field drainage. Avoid excessive nitrogen and unnecessary leaf wetness.",
            "pest": "For rice pests, inspect tillers and leaf surfaces regularly. Look for hopper damage, stem damage or caterpillar feeding and use integrated pest management where appropriate."
        },
        "maize": {
            "disease": "For maize, inspect leaves for expanding spots, streaks or blight-like lesions and check whether damage is concentrated on older or newer leaves.",
            "pest": "For maize, inspect the whorl for caterpillar feeding, frass and fresh damage. Early scouting is important for fall armyworm management."
        },
        "tomato": {
            "disease": "For tomato, inspect lower leaves first for dark spots, yellow halos or rapidly expanding lesions. Check humidity, leaf wetness and airflow.",
            "pest": "For tomato, inspect leaf undersides and growing tips for whiteflies, aphids, thrips and mites."
        },
        "chilli": {
            "disease": "For chilli, leaf curling can be caused by viral disease, sucking pests, heat or water stress. Check for whiteflies/thrips and distorted new growth before choosing treatment.",
            "pest": "For chilli, inspect young leaves and leaf undersides for thrips, aphids and whiteflies. Remove severely affected material where practical and monitor nearby plants."
        },
        "potato": {
            "disease": "For potato, inspect leaves for dark lesions and rapid spread, especially during cool and wet conditions. Avoid prolonged leaf wetness where possible.",
            "pest": "For potato, inspect leaves and stems for chewing damage, larvae and egg clusters."
        },
        "groundnut": {
            "disease": "For groundnut, check lower leaves for circular brown or dark spots and monitor whether symptoms are spreading upward.",
            "pest": "For groundnut, inspect leaves and young growth for chewing damage and insect activity."
        },
        "cotton": {
            "disease": "For cotton, inspect leaves, stems and bolls separately because nutrient, pest and disease symptoms can overlap.",
            "pest": "For cotton, inspect square and boll areas for caterpillar damage and scout regularly rather than spraying automatically."
        },
        "pulses": {
            "disease": "For pulses, check for wilting, vascular discoloration, root problems and field patterns. Remove severely affected plants where practical and improve field hygiene.",
            "pest": "For pulses, inspect flowers, pods and young leaves for insect feeding and monitor regularly."
        },
        "banana": {
            "disease": "For banana, inspect older leaves for streaks, spots and progressive leaf drying. Remove heavily affected leaves where appropriate and maintain field sanitation.",
            "pest": "For banana, inspect leaves, pseudostem and bunch area for visible insects or feeding damage."
        },
        "mango": {
            "disease": "For mango, inspect young leaves, flowers and fruit separately. Powdery growth on tender tissues needs different management from bacterial or fungal spotting.",
            "pest": "For mango, inspect tender shoots, flowers and fruit for sucking insects, hoppers and other pest activity."
        },
        "soybean": {
            "disease": "For soybean, inspect the lower canopy first for rust-like lesions or other leaf spots and monitor whether symptoms are increasing.",
            "pest": "For soybean, inspect leaves and pods for chewing or sucking damage and monitor the crop regularly."
        },
        "grape": {
            "disease": "For grape, inspect leaves and bunches for downy mildew-like lesions, powdery growth and moisture-related symptoms. Good canopy airflow helps reduce disease pressure.",
            "pest": "For grape, inspect leaves, shoots and bunches for mites, thrips and other visible pests."
        }
    }

    def farming_answer(question, crop, stage, duration, language="English"):
        q = question.lower().strip()
        crop_key = crop.lower() if crop != "Not specified" else ""
        context = []
        if crop != "Not specified":
            context.append(f"Crop: {crop}")
        if stage != "Not specified":
            context.append(f"Growth stage: {stage}")
        if duration != "Not specified":
            context.append(f"Duration: {duration}")

        context_text = " | ".join(context)
        crop_info = crop_guidance.get(crop_key, {})

        # Yellowing / nutrition / water stress
        if any(word in q for word in ["yellow", "yellowing", "pale", "chlorosis"]):
            answer = (
                "🌿 **Yellow-leaf check**\n\n"
                "Possible causes include water stress, nutrient imbalance, root problems, pests or disease.\n\n"
                "**Check these first:**\n"
                "1. Feel the soil near the root zone — is it very dry or waterlogged?\n"
                "2. Check whether yellowing starts on older leaves or new leaves.\n"
                "3. Inspect the underside of leaves for insects.\n"
                "4. Look for spots, streaks, curling or unusual patterns.\n"
                "5. Avoid adding fertilizer blindly before checking the likely cause.\n\n"
                "📷 If spots or unusual patterns are visible, use **AI Plant Doctor** for an image-based first screening."
            )

        # Pest questions
        elif any(word in q for word in ["pest", "insect", "aphid", "whitefly", "thrips", "caterpillar", "worm", "mite", "bug"]):
            answer = (
                "🐛 **Pest management checklist**\n\n"
                "1. Inspect the underside of leaves, young shoots, flowers and growing points.\n"
                "2. Look for eggs, larvae, webbing, sticky honeydew or chewing damage.\n"
                "3. Remove heavily damaged plant material where practical.\n"
                "4. Keep weeds and crop debris under control.\n"
                "5. Prefer monitoring and integrated pest management before chemical control.\n"
                "6. If a pesticide is needed, use only a product registered for the crop/pest and follow its label and local agricultural guidance.\n\n"
                "📷 A clear close-up photo of the pest can make identification easier."
            )
            if crop_info.get("pest"):
                answer += "\n\n🌾 **Crop-specific note:** " + crop_info["pest"]

        # Watering
        elif any(word in q for word in ["water", "watering", "irrigation", "dry", "wilting", "moisture"]):
            answer = (
                "💧 **Water management**\n\n"
                "• Check soil moisture before irrigating.\n"
                "• Avoid both prolonged dryness and standing water.\n"
                "• Apply water near the root zone rather than unnecessarily wetting leaves.\n"
                "• Consider crop stage, soil type, weather and recent rainfall.\n"
                "• If wilting continues even when soil is wet, inspect roots, pests and disease."
            )

        # Disease / spots
        elif any(word in q for word in ["disease", "spot", "rust", "blight", "mildew", "rot", "mosaic", "lesion", "fungus", "fungal"]):
            answer = (
                "🦠 **Disease first-response plan**\n\n"
                "1. Inspect several affected and healthy plants for comparison.\n"
                "2. Check whether symptoms are spreading and which plant part is affected first.\n"
                "3. Improve airflow and avoid unnecessary leaf wetness where possible.\n"
                "4. Remove infected debris where practical.\n"
                "5. Do not spray a chemical only because a leaf looks abnormal — symptoms can have several causes.\n"
                "6. Use **AI Plant Doctor** for an image-based first screening and confirm serious cases with a local agriculture expert."
            )
            if crop_info.get("disease"):
                answer += "\n\n🌾 **Crop-specific note:** " + crop_info["disease"]

        # Soil / fertilizer
        elif any(word in q for word in ["soil", "fertilizer", "nutrient", "manure", "nitrogen", "phosphorus", "potassium", "npk"]):
            answer = (
                "🌾 **Soil & nutrition guidance**\n\n"
                "• Prefer a soil test before making major fertilizer decisions.\n"
                "• Match nutrient management to crop and growth stage.\n"
                "• Maintain suitable organic matter and good drainage.\n"
                "• Avoid excessive fertilizer — more fertilizer does not always mean faster growth.\n"
                "• If symptoms are severe, compare affected and healthy plants before diagnosing a deficiency."
            )

        # Crop-specific general questions
        elif crop_info:
            answer = (
                f"🌱 **{crop} crop guidance**\n\n"
                f"For a useful first check, focus on four areas: soil moisture, pests, visible symptoms and growth stage.\n\n"
                f"🦠 **Disease:** {crop_info.get('disease', 'Inspect leaves, stems, roots and fruit for abnormal symptoms.')}\n\n"
                f"🐛 **Pests:** {crop_info.get('pest', 'Inspect young growth and leaf undersides regularly.')}\n\n"
                "📷 If your question is about a visible symptom, upload a clear image in **AI Plant Doctor**."
            )

        # Harvest / flowering / growth
        elif any(word in q for word in ["flower", "flowering", "fruit", "fruiting", "harvest", "yield", "growth"]):
            answer = (
                "🌼 **Growth-stage check**\n\n"
                "Crop needs change during seedling, vegetative, flowering, fruiting and maturity stages. "
                "Check moisture, nutrient balance, pest pressure and disease symptoms together rather than focusing on one factor.\n\n"
                "If you tell me the crop and growth stage, the advice can be made more specific."
            )

        # Generic but structured fallback
        else:
            answer = (
                "🤖 **AgriCare AI assessment**\n\n"
                "Start with these checks:\n"
                "1. 🌱 Crop and growth stage\n"
                "2. 💧 Soil moisture and recent irrigation/rain\n"
                "3. 🐛 Visible pests or insect damage\n"
                "4. 🦠 Leaf, stem, root or fruit symptoms\n"
                "5. 🌾 Recent fertilizer/manure application\n"
                "6. ☀️ Recent weather conditions\n\n"
                "📷 For a visible plant problem, use **AI Plant Doctor** with a clear image. "
                "AI output is an early screening aid, not a guaranteed field diagnosis."
            )

        if context_text:
            answer += f"\n\n📌 **Context used:** {context_text}"

        if duration != "Not specified" and duration in ["1–3 weeks", "More than 3 weeks"]:
            answer += "\n⏱️ Because the problem has persisted, compare affected vs. healthy plants and consider local expert verification."

        if language == "తెలుగు":
            return telugu_chat_answer(question, crop, stage, duration)

        return answer

    def telugu_chat_answer(question, crop, stage, duration):
        q = question.lower().strip()
        crop_text = crop if crop != "Not specified" else "మీ పంట"
        stage_text = stage if stage != "Not specified" else "పంట దశ"
        duration_text = duration if duration != "Not specified" else "వ్యవధి"

        if any(w in q for w in ["yellow", "yellowing", "pale", "chlorosis"]):
            return (f"🌿 **{crop_text} ఆకులు పసుపు రంగులోకి మారుతున్నాయా?**\n\n"
                    "సాధ్యమైన కారణాలు: నీటి ఒత్తిడి, పోషక లోపం, వేర్ల సమస్య, పురుగులు లేదా వ్యాధి.\n\n"
                    "**ముందుగా ఇవి పరిశీలించండి:**\n"
                    "1. వేర్ల దగ్గర నేల చాలా పొడిగా లేదా నీరు నిలిచిపోయిందా చూడండి.\n"
                    "2. పాత ఆకులా, కొత్త ఆకులా ముందుగా పసుపు అవుతున్నాయో చూడండి.\n"
                    "3. ఆకుల కింద పురుగులు, గుడ్లు లేదా జాలాలు ఉన్నాయా చూడండి.\n"
                    "4. మచ్చలు, ముడతలు, వంకరలు లేదా అసాధారణ ఆకారాలు ఉన్నాయా చూడండి.\n"
                    "5. కారణం తెలియకుండా ఎరువులు లేదా మందులు వేయవద్దు.\n\n"
                    "📷 కనిపించే లక్షణం ఉంటే **AI Plant Doctor**లో స్పష్టమైన ఆకుల ఫోటోను పరీక్షించండి.")
        if any(w in q for w in ["pest", "insect", "aphid", "whitefly", "thrips", "caterpillar", "worm", "mite", "bug"]):
            return (f"🐛 **{crop_text} పురుగు నియంత్రణ సూచనలు**\n\n"
                    "1. ఆకుల కింద భాగం, కొత్త కొమ్మలు, పువ్వులు మరియు పెరుగుతున్న భాగాలను పరిశీలించండి.\n"
                    "2. గుడ్లు, లార్వా, జాలాలు, తేనె వంటి అంటుకునే పదార్థం లేదా కొరికిన నష్టాన్ని చూడండి.\n"
                    "3. ఎక్కువగా దెబ్బతిన్న భాగాలను సాధ్యమైనంతవరకు తొలగించండి.\n"
                    "4. కలుపు మొక్కలు మరియు పంట అవశేషాలను నియంత్రించండి.\n"
                    "5. ముందుగా పర్యవేక్షణ మరియు సమగ్ర పురుగు నియంత్రణ (IPM) పద్ధతులను ఉపయోగించండి.\n"
                    "6. మందు అవసరమైతే ఆ పంట మరియు పురుగుకు స్థానికంగా నమోదు చేసిన మందును మాత్రమే లేబుల్ ప్రకారం వాడండి.\n\n"
                    "📷 పురుగు స్పష్టమైన క్లోజ్-అప్ ఫోటోను ఇవ్వడం ద్వారా గుర్తింపును మెరుగుపరచవచ్చు.")
        if any(w in q for w in ["water", "watering", "irrigation", "dry", "wilting", "moisture"]):
            return (f"💧 **{crop_text} నీటి నిర్వహణ**\n\n"
                    "• నీరు పెట్టే ముందు వేర్ల ప్రాంతంలోని నేల తేమను పరిశీలించండి.\n"
                    "• ఎక్కువ కాలం ఎండిపోవడం మరియు నీరు నిలవడం రెండింటినీ నివారించండి.\n"
                    "• సాధ్యమైనప్పుడు ఆకులపై కాకుండా వేర్ల ప్రాంతానికి నీరు ఇవ్వండి.\n"
                    f"• ప్రస్తుత పంట దశ: {stage_text}. వాతావరణం మరియు నేల రకాన్ని కూడా పరిగణించండి.\n"
                    "• నేల తడిగా ఉన్నప్పటికీ వాడిపోతే వేర్లు, పురుగులు మరియు వ్యాధిని పరిశీలించండి.")
        if any(w in q for w in ["disease", "spot", "rust", "blight", "mildew", "rot", "mosaic", "lesion", "fungus", "fungal"]):
            return (f"🦠 **{crop_text} వ్యాధి మొదటి చర్యలు**\n\n"
                    "1. ప్రభావిత మరియు ఆరోగ్యకరమైన మొక్కలను పోల్చి చూడండి.\n"
                    "2. లక్షణాలు వేగంగా వ్యాపిస్తున్నాయా, మొదట ఏ భాగంలో వచ్చాయో గమనించండి.\n"
                    "3. గాలి ప్రసరణ మెరుగుపరచండి మరియు అవసరం లేని ఆకుల తడిని తగ్గించండి.\n"
                    "4. సోకిన అవశేషాలను సాధ్యమైనంతవరకు తొలగించండి.\n"
                    "5. ఆకుల రూపాన్ని చూసి మాత్రమే రసాయన మందు పిచికారీ చేయవద్దు.\n"
                    "6. **AI Plant Doctor**లో ఫోటో పరీక్ష చేసి, తీవ్రమైన కేసులకు వ్యవసాయ నిపుణుడితో నిర్ధారించండి.\n\n"
                    f"📅 పంట దశ: {stage_text} | సమస్య వ్యవధి: {duration_text}")
        if any(w in q for w in ["soil", "fertilizer", "nutrient", "manure", "nitrogen", "phosphorus", "potassium", "npk"]):
            return (f"🌾 **{crop_text} నేల మరియు పోషక సూచనలు**\n\n"
                    "• పెద్ద ఎరువుల నిర్ణయం తీసుకునే ముందు నేల పరీక్ష చేయించుకోవడం మంచిది.\n"
                    "• పంట మరియు పెరుగుదల దశకు సరిపోయే పోషక నిర్వహణను పాటించండి.\n"
                    "• మంచి డ్రైనేజ్ మరియు తగిన సేంద్రియ పదార్థాన్ని నిర్వహించండి.\n"
                    "• అధిక ఎరువు వేయడం వల్ల ఎప్పుడూ మంచి దిగుబడి రాదు.\n"
                    "• లోపం అనుమానం ఉంటే ఆరోగ్యకరమైన మరియు ప్రభావిత మొక్కలను పోల్చండి.")
        return (f"🤖 **AgriCare AI — {crop_text} కోసం సూచన**\n\n"
                "మీ ప్రశ్నకు ఖచ్చితమైన సమాధానం ఇవ్వడానికి పంట, పెరుగుదల దశ, లక్షణాలు, నీటి పరిస్థితి మరియు సమస్య ఎంతకాలంగా ఉందో పరిశీలించాలి.\n\n"
                "📷 కనిపించే సమస్య అయితే స్పష్టమైన ఆకుల ఫోటోను **AI Plant Doctor**లో పరీక్షించండి.\n"
                "⚠️ AI సూచన ప్రాథమిక సహాయం మాత్రమే. రసాయన మందు వాడే ముందు స్థానికంగా నమోదు చేసిన ఉత్పత్తి లేబుల్ మరియు వ్యవసాయ నిపుణుల సలహాను పాటించండి.")

    # ---------------------------------------------------------
    # ASK + SIMPLE CHAT HISTORY
    # ---------------------------------------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    ask_col, clear_col = st.columns([4, 1])
    with ask_col:
        ask_clicked = st.button(
            "🤖 Ask AgriCare AI",
            use_container_width=True,
            type="primary",
            key="ask_farming_ai"
        )
    with clear_col:
        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
            key="clear_farming_chat"
        )

    if clear_clicked:
        st.session_state.chat_history = []
        st.rerun()

    if ask_clicked:
        if not user_question.strip():
            st.warning("Please enter a farming question.")
        else:
            response = farming_answer(
                user_question,
                chat_crop,
                chat_stage,
                chat_duration,
                chat_language
            )
            st.session_state.chat_history.append({
                "question": user_question.strip(),
                "answer": response
            })

    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🧑‍🌾 Your AI Assistance")
        for item in reversed(st.session_state.chat_history[-5:]):
            st.markdown("**You:** " + item["question"])
            st.success(item["answer"])
    else:
        st.caption("Your questions and AI responses will appear here.")

    # ---------------------------------------------------------
    # VOICE ASSISTANCE FOR AI CHAT
    # ---------------------------------------------------------
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("🔊 AI Chat Voice Assistance")
        voice_chat_lang = st.selectbox(
            "Voice language",
            ["English", "తెలుగు", "हिंदी"],
            index=["English", "తెలుగు", "हिंदी"].index(chat_language),
            key="chat_voice_language"
        )
        if st.button("🎧 Read latest AI answer aloud", use_container_width=True, key="chat_voice_button"):
            latest_answer = st.session_state.chat_history[-1]["answer"]
            # Remove markdown symbols for cleaner speech.
            speech_text = re.sub(r"[*_#`•]", "", latest_answer)
            speech_text = re.sub(r"\s+", " ", speech_text).strip()
            voice_code = {"English": "en", "తెలుగు": "te", "हिंदी": "hi"}[voice_chat_lang]
            try:
                with st.spinner("🔊 Creating AI voice..."):
                    tts = gTTS(text=speech_text, lang=voice_code, slow=False)
                    audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    tts.save(audio_file.name)
                    audio_file.close()
                st.audio(audio_file.name, format="audio/mp3")
            except Exception as e:
                st.error(f"Voice assistance is temporarily unavailable: {e}")

    # ---------------------------------------------------------
    # SAFETY / NEXT STEP
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 What makes a better AI answer?")
    a, b, c = st.columns(3)
    with a:
        st.info("🌱 **Crop**\nTell the crop name and variety if known.")
    with b:
        st.info("🔎 **Symptoms**\nTell where symptoms started and how they spread.")
    with c:
        st.info("📅 **History**\nMention watering, fertilizer and problem duration.")

    st.warning(
        "⚠️ AI assistance is for early guidance. Do not make high-risk pesticide or fertilizer decisions "
        "from AI output alone. Follow product labels and local agricultural recommendations. "
        "For serious or rapidly spreading problems, consult a qualified agriculture expert."
    )

    st.markdown("---")
    st.subheader("📚 Example Questions")
    examples = [
        "Why are my rice leaves turning yellow?",
        "My tomato has brown spots after rain. What should I check?",
        "How can I control pests in chilli?",
        "My maize leaves have holes. What should I inspect?",
        "How should I manage watering during flowering?",
        "What causes chilli leaf curling?",
        "How can I improve soil health?",
        "My crop problem has continued for two weeks. What should I do?"
    ]
    for example in examples:
        st.write("• " + example)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")


st.caption(
    "🌱 AgriCare AI | Early Detection and Management of "
    "Crop Diseases and Pest Infestations"
)
