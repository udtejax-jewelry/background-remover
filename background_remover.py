# -*- coding: utf-8 -*-
"""
Jewelry Background Remover — with optional Step-1 Gemini recolor preview.
Now includes:
- Authentication gate (allowed emails + shared password)
- Two tabs: "Background Remover" and "Jewelry Placer" (WIP)
"""

import io, os
import numpy as np
from PIL import Image
import streamlit as st

# ---- Load secrets safely ----
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
ALLOWED_EMAILS = set(e.lower() for e in st.secrets.get("ALLOWED_EMAILS", []))
SHARED_PASSWORD = st.secrets.get("SHARED_PASSWORD", "")
ALLOWED_DOMAINS = {e.split("@", 1)[1] for e in ALLOWED_EMAILS if "@" in e}

import cv2
from scipy import ndimage as ndi
from skimage.morphology import remove_small_objects
from skimage.filters import gaussian

# pymatting only needed if you choose closed_form mode
try:
    from pymatting.alpha.estimate_alpha_cf import estimate_alpha_cf
    HAS_PYMATTING = True
except Exception:
    HAS_PYMATTING = False

from rembg import remove, new_session


# =========================
# AUTHENTICATION
# =========================



def _normalize_email(email: str) -> str:
    """Lowercase entire email. Local-part case doesn't matter; domains must match exactly."""
    return (email or "").strip().lower()

def is_allowed_email(email: str) -> bool:
    email_n = _normalize_email(email)
    if "@" not in email_n:
        return False
    _, domain = email_n.split("@", 1)
    if domain not in ALLOWED_DOMAINS:
        return False
    return email_n in ALLOWED_EMAILS

def require_login():
    st.session_state.setdefault("auth_ok", False)
    st.session_state.setdefault("auth_email", "")

    if st.session_state["auth_ok"]:
        with st.sidebar:
            st.success(f"Signed in as {st.session_state['auth_email']}")
            if st.button("Sign out"):
                st.session_state["auth_ok"] = False
                st.session_state["auth_email"] = ""
                st.rerun()
        return  # user is authenticated

    st.title("🔒 Sign in")
    st.caption("Access restricted to approved emails and domains.")
    email = st.text_input("Work email", placeholder="you@tanyacreations.com")
    password = st.text_input("Password", type="password")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        submit = st.button("Sign in")

    if submit:
        if not is_allowed_email(email):
            st.error("Email not allowed. Please use an approved address with the exact domain.")
        elif password != SHARED_PASSWORD:
            st.error("Incorrect password.")
        else:
            st.session_state["auth_ok"] = True
            st.session_state["auth_email"] = _normalize_email(email)
            st.rerun()

    st.stop()  # block app until authenticated


# =========================
# Gemini client (for Step-1 preview) — secrets-first
# =========================
GEMINI_AVAILABLE = False
_gem_client = None
try:
    from google import genai
    from google.genai import types as gem_types  # noqa: F401 (kept for parity)

    # Prefer Streamlit Secrets; fallback to OS env for local dev
    _GEM_API_KEY = None
    try:
        _GEM_API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        _GEM_API_KEY = None
    if not _GEM_API_KEY:
        _GEM_API_KEY = os.environ.get("GEMINI_API_KEY")

    if _GEM_API_KEY:
        _gem_client = genai.Client(api_key=_GEM_API_KEY)
        GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False
    _gem_client = None

def gemini_recolor_background(pil_image: Image.Image, color_name: str) -> Image.Image | None:
    """
    Ask Gemini 2.5 Flash to replace ONLY the background with a uniform solid color.
    Preview use only; returns PIL image or None.
    """
    if not GEMINI_AVAILABLE:
        return None
    try:
        prompt = (
            f"Replace ONLY the background with a clean, uniform solid {color_name} color. "
            "Do not change the jewelry. No added text or borders. Output PNG."
        )
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        inp_img = Image.open(io.BytesIO(buf.getvalue()))

        resp = _gem_client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, inp_img],
        )
        for part in resp.candidates[0].content.parts:
            if getattr(part, "inline_data", None) is not None:
                out = Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                return out
        return None
    except Exception:
        return None


# =========================
# Rembg session
# =========================
@st.cache_resource
def _rembg_session(model_key: str = "isnet-general-use"):
    try:
        return new_session(model_key)
    except ValueError:
        for alt in ["isnet-general-use", "u2net", "u2netp", "u2net_human_seg"]:
            try:
                return new_session(alt)
            except Exception:
                continue
        raise


# =========================
# ---- Removal pipeline (UNCHANGED) ----
# =========================
def compose_rgba(image: Image.Image, alpha_float: np.ndarray) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    A = (np.clip(alpha_float, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb, A]), "RGBA")

def band_trimap_from_alpha(alpha_255: np.ndarray, band_px=3, conf_hi=220, conf_lo=35):
    a = alpha_255.astype(np.uint8)
    fg_conf = a >= conf_hi
    bg_conf = a <= conf_lo

    coarse = a >= 128
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*band_px+1, 2*band_px+1))
    dil = cv2.dilate(coarse.astype(np.uint8), k, 1).astype(bool)
    ero = cv2.erode(coarse.astype(np.uint8), k, 1).astype(bool)
    band = np.logical_xor(dil, ero)

    trimap = np.full_like(a, 128, np.uint8)
    trimap[bg_conf] = 0
    trimap[fg_conf] = 255
    trimap[band]    = 128
    return trimap

def guided_filter(I_gray: np.ndarray, p: np.ndarray, r: int, eps: float):
    I = I_gray.astype(np.float32)
    p = p.astype(np.float32)
    k = (2*r+1, 2*r+1)
    mean_I = cv2.boxFilter(I, -1, k)
    mean_p = cv2.boxFilter(p, -1, k)
    mean_Ip = cv2.boxFilter(I*p, -1, k)
    cov_Ip = mean_Ip - mean_I*mean_p
    mean_II = cv2.boxFilter(I*I, -1, k)
    var_I = mean_II - mean_I*mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    mean_a = cv2.boxFilter(a, -1, k)
    mean_b = cv2.boxFilter(b, -1, k)
    q = mean_a*I + mean_b
    return np.clip(q, 0.0, 1.0)

def refine_alpha(image_rgb: np.ndarray, alpha0: np.ndarray, trimap: np.ndarray,
                 mode: str, gf_radius: int, gf_eps: float):
    if mode == "none":
        return alpha0.astype(np.float32)/255.0
    if mode == "guided":
        base = alpha0.astype(np.float32)/255.0
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)/255.0
        refined = guided_filter(gray, base, r=max(1, gf_radius), eps=float(gf_eps))
        conf_fg = trimap == 255
        conf_bg = trimap == 0
        refined[conf_fg] = np.maximum(refined[conf_fg], base[conf_fg])
        refined[conf_bg] = np.minimum(refined[conf_bg], base[conf_bg])
        return np.clip(refined, 0.0, 1.0)
    if mode == "closed_form" and HAS_PYMATTING:
        I = image_rgb.astype(np.float64)/255.0
        T = trimap.astype(np.float64)/255.0
        T[T < 0.33] = 0.0
        T[(T>=0.33)&(T<=0.66)] = 0.5
        T[T > 0.66] = 1.0
        alpha = estimate_alpha_cf(I, T)
        return np.clip(alpha, 0.0, 1.0)
    return alpha0.astype(np.float32)/255.0

def choose_best_alpha(alpha0_255: np.ndarray, candidates: list[np.ndarray], trimap: np.ndarray):
    base = alpha0_255.astype(np.float32)/255.0
    best = base; best_score = -1e9
    conf_fg = trimap == 255; conf_bg = trimap == 0
    base_fg_area = (base >= 0.5).sum()
    for a in candidates:
        a = np.clip(a, 0.0, 1.0)
        fg_pen = np.abs(1.0 - a[conf_fg]).mean() if conf_fg.any() else 0.0
        bg_pen = np.abs(0.0 - a[conf_bg]).mean() if conf_bg.any() else 0.0
        penalty = fg_pen + bg_pen
        cand_fg_area = (a >= 0.5).sum()
        shrink = (base_fg_area - cand_fg_area) / max(1, base_fg_area)
        score = -penalty - 0.5*max(0.0, shrink)
        if score > best_score:
            best_score = score; best = a
    return best

def auto_hole_punch(alpha_float: np.ndarray, min_hole_area=60, smooth_px=0.8,
                    brightness_mask: np.ndarray | None = None, pearl_protect_level: float = 0.70):
    fg = alpha_float >= 0.5
    filled = ndi.binary_fill_holes(fg)
    holes = np.logical_and(filled, ~fg)
    holes = remove_small_objects(holes, min_size=int(max(0, min_hole_area)))
    if brightness_mask is not None:
        protect = brightness_mask >= pearl_protect_level
        holes[protect] = False
    if smooth_px and smooth_px > 0:
        holes = gaussian(holes.astype(float), sigma=float(smooth_px)) > 0.5
    out = alpha_float.copy()
    out[holes] = 0.0
    return out

def postprocess_alpha(alpha_float: np.ndarray, min_object_area=400, dehalo_px=0, edge_soften_px=1):
    mask = (alpha_float >= 0.5).astype(np.uint8)
    mask = remove_small_objects(mask.astype(bool), min_size=int(max(0, min_object_area))).astype(np.uint8)
    if dehalo_px and dehalo_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*dehalo_px+1, 2*dehalo_px+1))
        mask = cv2.erode(mask, k, 1)
    out = alpha_float.copy()
    out[mask == 0] = 0.0
    if edge_soften_px and edge_soften_px > 0:
        out = cv2.GaussianBlur(out.astype(np.float32), (0,0), float(edge_soften_px))
        out = np.clip(out, 0.0, 1.0)
    return out

def run_pipeline(
    image: Image.Image,
    session,
    band_px=3,
    conf_hi=220,
    conf_lo=35,
    refine_mode="guided",
    gf_radius=4,
    gf_eps=1e-3,
    min_hole_area=60,
    hole_smooth=0.8,
    pearl_protect_level=0.70,
    min_object_area=400,
    dehalo_px=0,
    edge_soften_px=1
):
    # initial alpha from rembg
    b = io.BytesIO(); image.save(b, format="PNG")
    rgba = remove(b.getvalue(), session=session, alpha_matting=False)
    pred = Image.open(io.BytesIO(rgba)).convert("RGBA")
    alpha0 = np.array(pred.split()[-1])  # 0..255

    img_rgb = np.array(image.convert("RGB"))

    # band trimap + refinement
    trimap = band_trimap_from_alpha(alpha0, band_px=band_px, conf_hi=conf_hi, conf_lo=conf_lo)

    cand_list = [alpha0.astype(np.float32)/255.0]
    cand_list.append(refine_alpha(img_rgb, alpha0, trimap, refine_mode, gf_radius, gf_eps))
    alpha = choose_best_alpha(alpha0, cand_list, trimap)

    # holes + cleanup
    gray_norm = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)/255.0
    alpha = auto_hole_punch(alpha, min_hole_area=min_hole_area,
                            smooth_px=hole_smooth, brightness_mask=gray_norm,
                            pearl_protect_level=pearl_protect_level)
    alpha_final = postprocess_alpha(alpha, min_object_area=min_object_area,
                                    dehalo_px=dehalo_px, edge_soften_px=edge_soften_px)

    return compose_rgba(image, alpha_final), (
        alpha0,
        (alpha*255).astype(np.uint8),
        (alpha_final*255).astype(np.uint8),
        trimap,
        (gray_norm*255).astype(np.uint8)
    )


# =========================
# UI ROOT (with auth + tabs)
# =========================
st.set_page_config(page_title="UD-NY Background Remover", page_icon="💎", layout="wide")

# Gate the whole app behind login
require_login()

st.title("💎 Jewelry Tools")

# Tabs
tab1, tab2 = st.tabs(["Background Remover", "Jewelry Placer"])

with tab2:
    st.header("Jewelry Placer")
    st.info("Work-In-Progress")

with tab1:

    with st.sidebar:
        st.header("Step-1: High-contrast background (Gemini)")
        enable_gemini = st.checkbox(
            "Use Gemini recolor pre-step",
            value=True if GEMINI_AVAILABLE else False,
            help="Preview only — background becomes a solid color to boost contrast.",
        )
        COLORS = [
            "teal", "pure black", "pure white", "electric blue", "deep navy", "royal purple",
            "bright green", "forest green", "crimson red", "bright orange", "mustard yellow",
            "charcoal gray"
        ]
        color_choice = st.selectbox("Solid color", COLORS, index=0)
        st.caption("Note: preview only; nothing is saved from Step-1.")

        st.header("Model (Step-2)")
        MODEL_OPTIONS = {
            "IS-Net (best)": "isnet-general-use",
            "U²-Net (classic)": "u2net",
            "U²-Net (lite)": "u2netp",
            "U²-Net (human)": "u2net_human_seg",
            "IS-Net (anime)": "isnet-anime",
        }
        model_label = st.selectbox("Rembg model", list(MODEL_OPTIONS.keys()), index=0)
        model_name = MODEL_OPTIONS[model_label]

        st.header("Trimap (from initial alpha)")
        band_px = st.slider("Unknown band width (px)", 1, 12, 3, 1)
        conf_hi = st.slider("Confident FG ≥", 160, 255, 220, 1)
        conf_lo = st.slider("Confident BG ≤", 0, 80, 35, 1)

        st.header("Refinement")
        refine_mode = st.selectbox("Mode", ["guided", "none", "closed_form"], index=0)
        gf_radius = st.slider("Guided radius (px)", 1, 15, 4, 1)
        gf_eps = st.select_slider(
            "Guided eps",
            options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2],
            value=1e-3,
        )

        st.header("Holes & Cleanup")
        min_hole_area = st.slider("Min hole area (px)", 0, 2000, 60, 10)
        hole_smooth = st.slider("Hole edge smoothing σ", 0.0, 3.0, 0.8, 0.1)
        pearl_protect_level = st.slider("Pearl protect brightness", 0.50, 0.95, 0.70, 0.01)
        min_object_area = st.slider("Min object area (px)", 0, 10000, 400, 50)
        dehalo_px = st.slider("Edge dehalo (px)", 0, 3, 0, 1)
        edge_soften_px = st.slider("Final edge softness (blur px)", 0, 4, 1, 1)

    session = _rembg_session(model_name)

    uploaded = st.file_uploader("Upload JPG/PNG", type=["jpg","jpeg","png"], accept_multiple_files=True)
    col1, col2, col3 = st.columns([1,1,1])

    if uploaded:
        for file in uploaded:
            orig = Image.open(file).convert("RGB")
            preview = None

            # --- Step-1: Gemini recolor (preview only)
            if enable_gemini:
                if GEMINI_AVAILABLE:
                    with st.spinner(f"Gemini: creating {color_choice} background preview..."):
                        preview = gemini_recolor_background(orig, color_choice)
                else:
                    st.warning("Gemini not available (missing google-genai or GEMINI_API_KEY). Skipping Step-1.")
            step1_image = preview if preview is not None else orig

            # show previews
            with col1:
                st.image(orig, caption=f"Input — {file.name}", use_container_width=True)
            with col2:
                if preview is not None:
                    st.image(preview, caption=f"Step-1 Preview ({color_choice})", use_container_width=True)
                else:
                    st.image(step1_image, caption="Step-1 Preview (skipped)", use_container_width=True)

            # --- Step-2: run UNCHANGED cutout pipeline on the Step-1 image
            out, debug = run_pipeline(
                image=step1_image, session=session,
                band_px=band_px, conf_hi=conf_hi, conf_lo=conf_lo,
                refine_mode=("closed_form" if (refine_mode=="closed_form" and HAS_PYMATTING) else refine_mode),
                gf_radius=gf_radius, gf_eps=float(gf_eps),
                min_hole_area=min_hole_area, hole_smooth=hole_smooth,
                pearl_protect_level=pearl_protect_level,
                min_object_area=min_object_area, dehalo_px=dehalo_px, edge_soften_px=edge_soften_px
            )

            with col3:
                st.image(out, caption="Step-2 Output (transparent PNG)", use_container_width=True)

            # Download final only
            buf = io.BytesIO(); out.save(buf, "PNG")
            st.download_button(
                "⬇️ Download final PNG",
                data=buf.getvalue(),
                file_name=os.path.splitext(file.name)[0]+"_cutout.png",
                mime="image/png",
            )

            with st.expander("Debug: initial / chosen / final / trimap / brightness"):
                a0, achosen, afinal, tri, bright = debug
                st.image(
                    [a0, achosen, afinal, tri, bright],
                    caption=["Initial alpha","Chosen alpha","Final alpha","Trimap","Brightness"],
                    use_container_width=True
                )
    else:
        st.info("Upload images. Use Step-1 Gemini to push the jewelry onto a strong-contrast solid color, then Step-2 cuts perfectly.")
