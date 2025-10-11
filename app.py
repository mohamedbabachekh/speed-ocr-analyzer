import io
import re
import json
from typing import Dict, Optional

import cv2
import numpy as np
from PIL import Image
import pytesseract
from flask import Flask, request, jsonify

app = Flask(__name__)

# -------- Helpers --------
DECIMAL_MB_PER_GB = 1000.0   # GB (decimal, used by most ISPs/Speedtest UI)
BINARY_MB_PER_GIB = 1024.0   # GiB (binary, IEC standard)

DL_LABELS = [
    r"descendant", r"download", r"downlink", r"dl"
]
UL_LABELS = [
    r"ascendant", r"upload", r"uplink", r"ul"
]

# e.g. "228", "66.5", "66,5"
NUM = r"(?P<num>\d+(?:[.,]\d+)?)"
MBPS = r"(?:\s*mbps|\s*m\s*bps|\s*megabits?/s?)?"

def to_float(s: str) -> float:
    return float(s.replace(",", "."))

def compute_units(mbps: float) -> Dict[str, float]:
    # Mbps → MB/s (divide by 8)
    MBps = mbps / 8.0
    # Decimal GB/h
    GB_per_h = MBps * 3600.0 / DECIMAL_MB_PER_GB
    # Binary GiB/h
    GiB_per_h = MBps * 3600.0 / BINARY_MB_PER_GIB
    return {
        "mbps": round(mbps, 3),
        "MBps": round(MBps, 3),
        "GB_per_hour_decimal": round(GB_per_h, 3),
        "GiB_per_hour_binary": round(GiB_per_h, 3),
    }

def preprocess_for_ocr(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # Gentle denoise + contrast
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=40, sigmaSpace=40)
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    # Adaptive threshold helps on glossy screens
    th = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10
    )
    return th

def ocr_text(img_bytes: bytes, lang: str = "eng+fra") -> str:
    # Read → preprocess → OCR
    file_bytes = np.frombuffer(img_bytes, np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        # as a fallback, try PIL directly
        text = pytesseract.image_to_string(Image.open(io.BytesIO(img_bytes)), lang=lang)
        return text

    pre = preprocess_for_ocr(bgr)
    pil = Image.fromarray(pre)
    text = pytesseract.image_to_string(pil, lang=lang)
    return text

def find_value(text: str, labels: list) -> Optional[float]:
    """
    Look for a label (e.g., 'Descendant' / 'Downlink') and capture the number
    near it. We search both 'label ... number' and 'number ... label' patterns.
    """
    t = text.lower()
    # Normalize “mb/s” weird spacing
    t = re.sub(r"m\s*bps", "mbps", t)

    label_pat = r"|".join(labels)

    patterns = [
        rf"(?:{label_pat}).{{0,25}}{NUM}\s*{MBPS}",
        rf"{NUM}\s*{MBPS}.{{0,25}}(?:{label_pat})",
    ]
    for pat in patterns:
        m = re.search(pat, t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            try:
                return to_float(m.group("num"))
            except Exception:
                pass
    # As a fallback, capture big numbers on the line that contains the label
    for label in labels:
        for line in t.splitlines():
            if label in line:
                m = re.search(NUM, line)
                if m:
                    try:
                        return to_float(m.group("num"))
                    except Exception:
                        continue
    return None

# -------- API --------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Multipart form-data: file=@your_image.jpg
    Optional query/form: lang=eng+fra+ara  (if you also want Arabic)
    """
    if "file" not in request.files:
        return jsonify({"error": "Upload with form field 'file'"}), 400

    lang = request.form.get("lang", "eng+fra")
    img = request.files["file"].read()
    text = ocr_text(img, lang=lang)

    dl_mbps = find_value(text, DL_LABELS)
    ul_mbps = find_value(text, UL_LABELS)

    result = {
        "raw_ocr_text": text,
        "parsed": {}
    }

    if dl_mbps is not None:
        result["parsed"]["downlink"] = compute_units(dl_mbps)
    if ul_mbps is not None:
        result["parsed"]["uplink"] = compute_units(ul_mbps)

    # Basic validation
    if not result["parsed"]:
        return jsonify({
            "warning": "Could not confidently find downlink/uplink Mbps; check OCR text.",
            "raw_ocr_text": text
        }), 422

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)

