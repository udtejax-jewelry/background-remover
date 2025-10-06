# 💎 Jewelry Tools (Gemini + Rembg)

A dual-purpose **Streamlit app** for jewelry image workflows:

1. **Background Remover**
   - Optional **Step-1:** Uses **Gemini 2.5 Flash** to replace the background with a solid, high-contrast color (preview only).
   - **Step-2:** Runs a robust cutout pipeline (Rembg + band-trimap + guided/closed-form matting + pearl-safe cleanup) to produce a clean **transparent PNG**.

2. **Jewelry Placer** *(Work-In-Progress)*
   - Placeholder tab for upcoming features.

The app is access-controlled via a lightweight email/password authentication system.

---

## ✨ Features

- 🪄 **High-contrast background preview** powered by Gemini 2.5 Flash.
- 🎯 **Accurate cutouts** with IS-Net / U²-Net models via Rembg.
- 🔍 **Refinement controls:** guided or closed-form matting.
- ⚪ **Pearl-safe hole punch** — protects bright beads or hollow loops.
- ⚙️ Interactive sliders for trimap, smoothing, dehalo, and edge softness.
- 🖼️ Batch upload, preview, and transparent PNG download.
- 🔐 Authentication via whitelisted company emails + shared password.

---

## 🗂️ Project Structure

---

## 🔐 Authentication

> ⚠️ *This is a simple in-app login intended for internal/company use only.  
> For production security, switch to OAuth, SSO, or Streamlit Teams access control.*

---

## 🧰 Requirements

Add the following to your `requirements.txt`:

altair==5.5.0
annotated-types==0.7.0
anyio==4.10.0
attrs==25.3.0
blinker==1.9.0
cachetools==5.5.2
certifi==2025.8.3
charset-normalizer==3.4.3
click==8.2.1
colorama==0.4.6
coloredlogs==15.0.1
flatbuffers==25.2.10
gitdb==4.0.12
GitPython==3.1.45
google-auth==2.40.3
google-genai==1.38.0
h11==0.16.0
httpcore==1.0.9
httpx==0.28.1
humanfriendly==10.0
idna==3.10
imageio==2.37.0
Jinja2==3.1.6
jsonschema==4.25.1
jsonschema-specifications==2025.9.1
lazy_loader==0.4
llvmlite==0.45.0
MarkupSafe==3.0.2
mpmath==1.3.0
narwhals==2.5.0
networkx==3.5
numba==0.62.0
numpy==2.2.6
onnxruntime==1.23.0
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
packaging==25.0
pandas==2.3.2
pillow==11.3.0
platformdirs==4.4.0
pooch==1.8.2
protobuf==6.32.1
pyarrow==21.0.0
pyasn1==0.6.1
pyasn1_modules==0.4.2
pydantic==2.11.9
pydantic_core==2.33.2
pydeck==0.9.1
PyMatting==1.1.12
pyreadline3==3.5.4
python-dateutil==2.9.0.post0
python-dotenv==1.1.1
pytz==2025.2
referencing==0.36.2
rembg==2.0.67
requests==2.32.5
rpds-py==0.27.1
rsa==4.9.1
scikit-image==0.25.2
scipy==1.16.2
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
streamlit==1.49.1
sympy==1.14.0
tenacity==9.1.2
tifffile==2025.9.20
toml==0.10.2
tornado==6.5.2
tqdm==4.67.1
typing-inspection==0.4.1
typing_extensions==4.15.0
tzdata==2025.2
urllib3==2.5.0
watchdog==6.0.0
websockets==15.0.1


---

## 🔑 Environment Variables

### Local development

Create a `.env` file in your project root:

Without the key, the app still works (Gemini pre-step will be skipped).

### Streamlit Community Cloud

Go to **App → Settings → Secrets**, then add:

```toml
GEMINI_API_KEY = "your_google_gemini_key"


# 1️⃣ Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ (Optional) Add GEMINI_API_KEY to .env

# 4️⃣ Launch the app
streamlit run background_remover.py

background_remover_v2.py

GEMINI_API_KEY="your_google_gemini_key"

### 🧭 How to Use

Log in with an approved company email and password.

Go to the Background Remover tab.

Upload one or more jewelry photos (.jpg or .png).

(Optional) Enable the Gemini pre-step and select a background color that contrasts the jewelry.

Adjust sliders only if needed:

Trimap band width

Refinement mode (guided recommended)

Hole area, pearl protect level, dehalo, edge softness

Download the final transparent PNG.

Check the Jewelry Placer tab — currently marked as Work-In-Progress.

### 🧪 Recommended Settings
Setting	Recommended
Refinement	guided
Band width	3–6 px
Pearl protect	0.70–0.85
Hole area	40–80 px
Edge softness	1 px
Dehalo	0

If edges look eroded or jewelry is lost, use Gemini pre-step for a high-contrast color and stick with guided refinement.

### 🛠️ Troubleshooting

Gemini pre-step not working

Ensure your GEMINI_API_KEY is correctly set in .env or Streamlit Secrets.

ONNX Runtime Error

Use onnxruntime==1.18.0 or install onnxruntime-gpu if on CUDA.

Jewelry too thin or missing details

Set Refinement to guided or none, increase Band Width, and set Dehalo to 0.

Inner holes not visible

Lower Min hole area or increase Hole edge smoothing σ.

### 🔒 Security Notice

This app uses simple, static login credentials for internal testing.
For production deployment, replace with an enterprise-grade authentication system (OAuth / SSO) and store credentials in Streamlit Secrets.

### 🧭 Roadmap

✅ Background Remover (complete)

🚧 Jewelry Placer (in progress)

🔜 Preset profiles for jewelry styles (rings, chains, pendants)

🔜 Manual mask refinement (scribble mode)

🔜 Organization-wide SSO auth

### 🤝 Contributing

Pull requests welcome!
Please:

Keep the background removal logic intact unless fixing bugs.

Add new features under the appropriate tab.

Include documentation updates when adding new settings or dependencies.

### 📜 License

Copyright © 2025
All rights reserved.

If you plan to open-source, replace this with an open-source license (e.g., MIT or Apache 2.0).

### 🙏 Acknowledgements

Rembg
 — background removal backend

PyMatting
 — closed-form alpha matting

Google Gemini 2.5 Flash
 — background recolor step

Streamlit
 — app framework powering this project

### 🏁 Summary

This app combines AI-powered background replacement with classic matting techniques, offering a reliable jewelry cutout pipeline that even handles pearls, loops, and soft edges — all inside an intuitive Streamlit interface.


