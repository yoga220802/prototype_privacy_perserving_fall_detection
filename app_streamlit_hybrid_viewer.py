import os

import requests
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Privacy-Preserving Fall Detection - Hybrid Streamlit",
    layout="wide",
)

API_INTERNAL_URL = os.getenv("API_INTERNAL_URL", "http://fall-detection-gpu-api:8000")
BRIDGE_INTERNAL_URL = os.getenv("BRIDGE_INTERNAL_URL", "http://host.docker.internal:7001")
BRIDGE_BROWSER_URL = os.getenv("BRIDGE_BROWSER_URL", "http://localhost:7001")

st.title("🤸‍♂️ Privacy-Preserving Fall Detection - Hybrid Streamlit")

st.info(
    "Kamera dibuka native di Windows lewat OpenCV CAP_DSHOW, inferensi berjalan di Docker GPU, "
    "dan hasilnya ditampilkan kembali di halaman Streamlit ini sebagai MJPEG stream."
)

col_status_1, col_status_2 = st.columns(2)

with col_status_1:
    st.subheader("Docker GPU API")
    try:
        r = requests.get(f"{API_INTERNAL_URL}/health", timeout=3)
        if r.ok:
            data = r.json()
            if data.get("cuda_available"):
                st.success(f"CUDA aktif: {data.get('gpu')}")
            else:
                st.warning("CUDA tidak aktif")
            st.json(data)
        else:
            st.error(f"API health gagal: {r.status_code}")
    except Exception as exc:
        st.error(f"Tidak bisa menghubungi API dari Streamlit container: {exc}")

with col_status_2:
    st.subheader("Native Camera Bridge")
    try:
        r = requests.get(f"{BRIDGE_INTERNAL_URL}/status", timeout=3)
        if r.ok:
            data = r.json()
            if data.get("camera_opened"):
                st.success(
                    f"Kamera aktif | FPS {data.get('camera_fps')} | API {data.get('api_latency_ms')} ms"
                )
            else:
                st.warning("Bridge aktif, tetapi kamera belum terbuka")
            st.json(data)
        else:
            st.error(f"Bridge status gagal: {r.status_code}")
    except Exception as exc:
        st.error(
            "Native Camera Bridge belum jalan. Jalankan host_camera_mjpeg_bridge.py di Windows native.\n\n"
            f"Detail: {exc}"
        )

st.divider()

raw_url = f"{BRIDGE_BROWSER_URL}/raw.mjpg"
result_url = f"{BRIDGE_BROWSER_URL}/result.mjpg"

left, right = st.columns([1, 1])

with left:
    st.subheader("1. Native Windows Camera - RAW")
    components.html(
        f"""
        <div style="width:100%; background:#111; padding:8px; border-radius:12px;">
            <img src="{raw_url}" style="width:100%; height:auto; border-radius:8px;" />
        </div>
        """,
        height=520,
    )

with right:
    st.subheader("2. Docker GPU Result")
    components.html(
        f"""
        <div style="width:100%; background:#111; padding:8px; border-radius:12px;">
            <img src="{result_url}" style="width:100%; height:auto; border-radius:8px;" />
        </div>
        """,
        height=520,
    )

st.divider()

st.subheader("Cara menjalankan")
st.code(
    r"""
# Terminal 1: jalankan Docker API + Streamlit Viewer
docker compose -f docker-compose.hybrid.streamlit.yml up --build

# Terminal 2: jalankan kamera native Windows
.\.venv-camera\Scripts\Activate.ps1
python host_camera_mjpeg_bridge.py --camera 0 --width 1280 --height 720 --fps 30 --process-every 3 --target-size 640 --view triple
""",
    language="powershell",
)

st.caption(
    "Halaman ini tidak memproses frame di Streamlit. Streamlit hanya menampilkan MJPEG stream, "
    "sehingga kualitas kamera jauh lebih dekat ke native Windows dibanding WebRTC Streamlit."
)
