Jewelry Tools (Gemini + Rembg)

Two-in-one Streamlit app for jewelry image workflows:

Background Remover

Optional Step-1: uses Gemini 2.5 Flash to place the jewelry on a solid, high-contrast background (preview only).

Step-2: runs a robust cutout pipeline (Rembg + band-trimap + guided/CF matting + pearl-safe hole punch) to produce a clean transparent PNG.

Jewelry Placer (WIP)

Placeholder tab for future features.

The app is gated by a lightweight email/password login.

✨ Features

High-contrast pre-step (Gemini) for near “one-click” results on tricky metals/pearls.

Rembg (IS-Net/U²-Net family) for strong initial alpha.

Band-trimap refinement with Guided Filter (safe) or Closed-Form Matting (precise).

Pearl-safe hole punch: opens loops/ball centers without deleting bright beads.

Interactive sliders for thresholds, hole size, edge dehalo, and softness.

Batch upload, preview, and download final transparent PNGs.

Simple Auth: whitelisted emails + shared password.

🗂 Project Structure
.
├─ background_remover.py     # Streamlit app (includes auth + both tabs)
├─ requirements.txt          # Python dependencies
├─ .env                      # (optional) holds GEMINI_API_KEY for local dev
└─ README.md

🔐 Authentication

Allowed emails (case-insensitive local part, strict domain match)

Note: This is an app-level gate intended for internal usage. For sensitive deployments, switch to a proper auth solution (OAuth/SSO, Streamlit Teams access control, etc.).

🧰 Requirements

See requirements.txt:

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


🔑 Environment Variables
Local development

Create a .env file in the project root:

GEMINI_API_KEY=your_google_gemini_key


Without this key, the app still runs; the Gemini pre-step is simply skipped.

Streamlit Community Cloud

Add the key under App → Settings → Secrets:

GEMINI_API_KEY = "your_google_gemini_key"

▶️ Run Locally
# 1) create & activate a venv (recommended)
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) install deps
pip install -r requirements.txt

# 3) (optional) add GEMINI_API_KEY to .env

# 4) launch
streamlit run background_remover.py


Open the URL printed in the terminal. You’ll see the login screen first.

🚀 Deploy to Streamlit Community Cloud

Push this repo to GitHub.

In Streamlit Cloud, New app → select the repo & background_remover.py.

Under Advanced settings → Secrets, add:

GEMINI_API_KEY="your_google_gemini_key"


Deploy.

Share the app link with the approved users only.

🧭 Usage

Sign in with an allowed email + password.

Background Remover tab:

Upload one or more JPG/PNG images.

(Optional) Enable Gemini pre-step and pick a color that provides the strongest contrast.

Tune sliders only if needed:

Trimap band width, refinement mode (guided recommended), hole size/smoothing, pearl protect, dehalo, edge softness.

Download the final transparent PNG.

Jewelry Placer tab: shows Work-In-Progress (placeholder).

🧪 Recommended Settings

Start with defaults:

Refinement: guided

Band width: 3–6 px

Dehalo: 0

Pearl protect: 0.70–0.85 if beads get nibbled

Min hole area: 40–80 depending on piece size

If edges erode on bright backgrounds, rely on Gemini pre-step and keep guided mode.

🛠 Troubleshooting

Gemini step not working

Make sure GEMINI_API_KEY is set. If missing, the app will skip the pre-step and show a warning.

ONNX runtime errors

Ensure onnxruntime==1.18.0 (or onnxruntime-gpu on CUDA setups).

Cutout too aggressive / missing details

Set Refinement to guided or none; increase band width; set Dehalo to 0; raise Pearl protect.

Inner holes not transparent

Lower Min hole area or increase Hole edge smoothing slightly.

🔒 Security Notes

Current auth is a simple email whitelist + shared password. Avoid storing sensitive data.

For production-grade security, replace with SSO/OAuth or Streamlit Teams access control and unique user credentials.

🧭 Roadmap

Finish Jewelry Placer features.

Preset profiles per jewelry style (bracelet, keychain, charm).

Optional manual scribble refinement (GrabCut) for edge cases.

Proper org SSO auth.

🤝 Contributing

PRs welcome! Please:

Keep the removal pipeline logic intact unless fixing a bug.

Put new UI features under the appropriate tab.

Use clear commit messages and update this README if behavior changes.

📄 License

Copyright © 2025. All rights reserved.
If you plan to open-source, replace with MIT/Apache-2.0 and add a LICENSE file.

🙏 Acknowledgements

Rembg
 for high-quality background removal.

pymatting
 for closed-form matting.

Google Gemini 2.5 Flash for the high-contrast background preview step.

Streamlit for the app framework.

Questions or feature requests? Open an issue in this repo.
