import os
import io
import re
import base64,json
import pandas as pd
import streamlit as st
from PIL import Image
from typing import Dict, Optional

# OpenAI SDK (Responses API)
# from dotenv import load_dotenv
import os
from openai import OpenAI

# load_dotenv()  # Loads .env file
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ------------- Streamlit UI -------------
st.set_page_config(page_title="📡 Speed OCR Analyzer (OpenAI)", page_icon="📶", layout="wide")
st.title("📡 Speed OCR Analyzer (OpenAI Vision)")
st.caption("Upload one or more Speedtest screenshots. The app extracts Downlink/Uplink Mbps using OpenAI Vision, then computes MB/s and GB/h.")

# ------------- Sidebar Configuration -------------
# ------------- Sidebar Configuration -------------
with st.sidebar:
    st.header("⚙️ Settings")

    # Model selector
    model = st.selectbox("Model", ["gpt-4o", "gpt-4o-mini"], index=0)
    st.markdown("**Note:** Images are sent to OpenAI for analysis.")

    # Paste API key
    api_key_input = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="Paste your OpenAI API key here..."
    )

    # Determine which key to use
    if api_key_input.strip():
        api_key = api_key_input.strip()
        st.success("✅ Using API key from input box")
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
        st.success("✅ Using API key from environment variable")
    else:
        api_key = None
        st.error("❌ No API key found. Please paste one above or set it as an environment variable.")

    st.divider()
    st.caption("You can obtain an API key from [platform.openai.com](https://platform.openai.com/api-keys).")

# Initialize client once (only if key exists)
client: Optional[OpenAI] = OpenAI(api_key=api_key) if api_key else None


# ------------------ HELPERS ------------------
DECIMAL_MB_PER_GB = 1000.0
BINARY_MB_PER_GIB = 1024.0

def compute_units(mbps: float) -> Dict[str, float]:
    """Compute MB/s, GB/h, and GiB/h from Mbps."""
    if not isinstance(mbps, (int, float)):
        return {}
    MBps = mbps / 8.0
    GB_per_h = MBps * 3600.0 / DECIMAL_MB_PER_GB
    GiB_per_h = MBps * 3600.0 / BINARY_MB_PER_GIB
    return {
        "mbps": round(mbps, 3),
        "MBps": round(MBps, 3),
        "GB_per_hour_decimal": round(GB_per_h, 3),
        "GiB_per_hour_binary": round(GiB_per_h, 3),
    }

def to_b64(img_bytes: bytes, mime="image/jpeg") -> str:
    return f"data:{mime};base64," + base64.b64encode(img_bytes).decode("utf-8")


# Strict schema for structured output
SPEEDTEST_SCHEMA = {
    "name": "speedtest_reading",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "downlink_mbps": {"type": ["number", "null"], "description": "Download speed in Mbps"},
            "uplink_mbps": {"type": ["number", "null"], "description": "Upload speed in Mbps"},
            "ping_ms": {"type": ["number", "null"], "description": "Latency in milliseconds"},
            "test_time": {"type": ["string", "null"], "description": "Clock time at top of screen, e.g. '20:49'"},
            "device_info": {"type": ["string", "null"], "description": "Device and operator info, e.g. 'inwi SM-S906E | Orange Casablanca'"},
            "notes": {"type": "string", "description": "Any detection remarks or confidence comments"}
        },
        "required": ["downlink_mbps", "uplink_mbps"]
    }
}

SYSTEM_INSTRUCTIONS = (
    "You are an OCR assistant for Speedtest screenshots. "
    "Extract numeric values for download (descendant/downlink) and upload (ascendant/uplink) speeds in Mbps, "
    "and ping latency (ms) if visible. Also extract the visible clock time and device/operator info. "
    "Normalize commas to periods (e.g., 66,5 → 66.5). Return only JSON conforming to the schema. "
    "If a field is not visible, return null."
)

def ask_openai_for_speeds(img_bytes: bytes, model_name: str) -> Dict:
    """Send image to OpenAI Vision model; parse numeric values manually if structured output not supported."""
    if not client:
        return {"notes": "❌ No OpenAI client initialized"}

    data_url = to_b64(img_bytes, mime="image/jpeg")

    try:
        resp = client.responses.create(
            model=model_name,
            instructions=SYSTEM_INSTRUCTIONS,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text",
                     "text": (
                         "Extract Downlink (Mbps), Uplink (Mbps), Ping (ms), visible clock time (e.g. '20:49'), "
                         "and device/operator info (e.g. 'inwi SM-S906E | Orange Casablanca'). "
                         "Return valid JSON with keys: downlink_mbps, uplink_mbps, ping_ms, test_time, device_info, notes."
                     )},
                    {"type": "input_image", "image_url": data_url}
                ],
            }],
        )

        raw = getattr(resp, "output_text", "")
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))

        # fallback: regex detection
        text = raw.lower()
        dl = ul = ping = None
        time_str = None
        device_info = None

        num = r"(\d+(?:[.,]\d+)?)"
        dl_m = re.search(r"(descendant|download|downlink|dl)\D{0,10}" + num, text, re.I)
        ul_m = re.search(r"(ascendant|upload|uplink|ul)\D{0,10}" + num, text, re.I)
        p_m = re.search(r"(ping)\D{0,6}" + num, text, re.I)
        t_m = re.search(r"\b(\d{1,2}[:hH]\d{2})\b", text)
        op_m = re.search(r"(inwi|maroc ?telecom|orange).*?(casablanca|rabat|tanger|fes|marrakech)?", text, re.I)

        if dl_m: dl = float(dl_m.group(2).replace(",", "."))
        if ul_m: ul = float(ul_m.group(2).replace(",", "."))
        if p_m: ping = float(p_m.group(2).replace(",", "."))
        if t_m: time_str = t_m.group(1).replace("h", ":").replace("H", ":")
        if op_m: device_info = op_m.group(0).strip()

        return {
            "downlink_mbps": dl,
            "uplink_mbps": ul,
            "ping_ms": ping,
            "test_time": time_str,
            "device_info": device_info,
            "notes": "Parsed via fallback regex"
        }

    except Exception as e:
        return {
            "downlink_mbps": None,
            "uplink_mbps": None,
            "ping_ms": None,
            "test_time": None,
            "device_info": None,
            "notes": f"API error: {e}"
        }


# ------------------ STREAMLIT APP ------------------
# st.title("📡 Speedtest Analyzer (AI Vision)")
# st.caption("Upload Speedtest screenshots — extract Mbps, Ping, Time, and Device/Operator automatically.")

uploaded_files = st.file_uploader(
    "📸 Upload one or more Speedtest screenshots",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    rows = []
    st.info(f"Processing {len(uploaded_files)} image(s) using {model}...")
    progress = st.progress(0)

    for i, f in enumerate(uploaded_files, start=1):
        name = f.name
        img_bytes = f.getvalue()

        with st.expander(f"Preview — {name}"):
            st.image(Image.open(io.BytesIO(img_bytes)), caption=name, width=512)

        result = ask_openai_for_speeds(img_bytes, model)
        dl, ul, ping = result.get("downlink_mbps"), result.get("uplink_mbps"), result.get("ping_ms")
        time_str, device_info, notes = result.get("test_time"), result.get("device_info"), result.get("notes")

        down = compute_units(dl)
        up = compute_units(ul)

        rows.append({
            "Image": name,
            "Downlink (Mbps)": dl,
            "Uplink (Mbps)": ul,
            "Ping (ms)": ping,
            "Downlink MB/s": down.get("MBps"),
            "Uplink MB/s": up.get("MBps"),
            "Test Time": time_str,
            "Device / Operator": device_info,
            "Notes": notes,
        })

        progress.progress(i / len(uploaded_files))

    df = pd.DataFrame(rows)
    st.success("✅ Analysis complete")

    st.subheader("📊 Summary Table")
    st.dataframe(
        df[["Image", "Downlink (Mbps)", "Uplink (Mbps)", "Ping (ms)",
            "Downlink MB/s", "Uplink MB/s", "Test Time", "Device / Operator", "Notes"]],
        width='stretch'
    )

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download results as CSV",
        data=csv,
        file_name="speedtest_ai_results.csv",
        mime="text/csv"
    )
else:
    st.info("👆 Upload one or more images to start the analysis.")