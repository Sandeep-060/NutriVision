from __future__ import annotations
import base64
import hashlib
import io
import json
import logging
import sys
from html import escape
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# --------------------------------------------------------------------------
# Paths + import path
#   Project root = parent of app/. It must be importable so that
#   `from src.segmentation... import ...` works no matter how Streamlit is launched.
# --------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.nutrition.calculator import calculate_meal_nutrition, load_nutrition_reference

HERO_IMAGE_PATH = APP_DIR / "assets" / "hero_image.png"
SAMPLE_IMAGE_PATH = APP_DIR / "assets" / "sample_meal.png"
CSS_PATH = APP_DIR / "style.css"
FOODSEG_LABELS_PATH = PROJECT_ROOT / "data" / "raw" / "FoodSeg103" / "id2label.json"

logger = logging.getLogger("nutrivision")

st.set_page_config(page_title="NutriVision", page_icon="\U0001F957", layout="wide",
                   initial_sidebar_state="collapsed")

# --------------------------------------------------------------------------
# Real model / FoodSeg103 configuration
# --------------------------------------------------------------------------
NUM_FOOD_CLASSES = 103          # class 0 = background, 1..103 = FoodSeg103
PREVIEW_MAX_SIDE = 1024         # preview only; inference always uses the original bytes


@st.cache_data(show_spinner=False)
def load_foodseg_labels() -> dict[int, str]:
    """FoodSeg103 class id -> canonical name (background / id 0 excluded)."""
    if not FOODSEG_LABELS_PATH.exists():
        raise FileNotFoundError(f"FoodSeg103 label mapping not found: {FOODSEG_LABELS_PATH}")
    with FOODSEG_LABELS_PATH.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    items = raw.items() if isinstance(raw, dict) else enumerate(raw)
    labels = {int(k): str(v).strip() for k, v in items if int(k) != 0}
    if len(labels) != NUM_FOOD_CLASSES:
        raise ValueError(f"Expected {NUM_FOOD_CLASSES} FoodSeg103 food classes, found {len(labels)}.")
    return labels


@st.cache_data(show_spinner=False)
def get_nutrition_reference():
    """Stage 7 nutrition reference (source of truth), loaded once via calculator.py."""
    return load_nutrition_reference()


@st.cache_resource(show_spinner="Loading NutriVision model\u2026")
def get_model():
    """Load the Keras SegFormer once per Streamlit process and reuse it on every rerun.

    The spinner text above is only shown on a cache MISS (i.e. the first analysis);
    afterwards this returns the cached model instantly. Exceptions are not cached,
    so a failed load can be retried.
    """
    from src.segmentation.model import build_model, MODEL_PATH   # lazy: keeps Home fast
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH} (launch Streamlit from the project root: "
            f"streamlit run app/app.py)")
    return build_model()


HOW_STEPS = [
    ("Snap", "Upload a photo",
     "Take or upload one photo of your meal, exactly as it looks on the plate."),
    ("Spot", "We find each food",
     "NutriVision scans the photo and marks every food item it can identify, one by one."),
    ("Check", "You confirm it's right",
     "Quickly confirm, correct, or skip anything NutriVision got wrong \u2014 you always have the final say."),
    ("Total", "Get your nutrition",
     "Enter how much you ate and get calories, protein, carbs, fat and fiber for the full meal."),
]

STEP_NAMES = ["1 \u00b7 Upload", "2 \u00b7 Review", "3 \u00b7 Quantity", "4 \u00b7 Results"]


# --------------------------------------------------------------------------
# Small helpers: HTML, CSS, images
# --------------------------------------------------------------------------
def html(markup: str) -> None:
    """Render raw HTML. Lines are flattened so Markdown never treats indentation as a code block."""
    flat = " ".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)


def load_css() -> None:
    if CSS_PATH.exists():
        st.markdown(f"<style>{CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def file_to_data_uri(path_str: str) -> str | None:
    path = Path(path_str)
    if not path.exists():
        return None
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def decode_image(image_bytes: bytes) -> Image.Image:
    """The single decode path for both preview and inference (EXIF-corrected, RGB)."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ValueError("That file couldn't be read as an image. Please use a JPG or PNG.") from exc


def image_bytes_to_data_uri(image_bytes: bytes) -> str:
    """Optimised, aspect-preserving preview. Used for DISPLAY only - never for inference."""
    image = decode_image(image_bytes)
    image.thumbnail((PREVIEW_MAX_SIDE, PREVIEW_MAX_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Session state + callbacks
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "screen": "home",       # "home" | "app"
        "step": 1,              # 1..4 (inside the app)
        "max_reached": 1,
        "img_uri": None,        # optimised preview (display only)
        "image_bytes": None,    # ORIGINAL bytes (used for inference)
        "image_key": None,      # sha1 of image_bytes
        "analyzed_key": None,   # image_key that `detections` belongs to
        "analyzing": False,
        "upload_sig": None,
        "detections": [],
        "detection_error": None,
        "quantities": {},
        "qty_errors": [],
        "nutrition": None,          # result of calculate_meal_nutrition() for the current meal
        "nutrition_error": None,
        "uploader_n": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_analysis() -> None:
    """Forget everything derived from the previous image (detections, decisions, quantities)."""
    ss = st.session_state
    for key in list(ss.keys()):
        if key.startswith(("sel_", "qty_")):
            del ss[key]
    ss.update(detections=[], quantities={}, qty_errors=[], nutrition=None, nutrition_error=None,
              analyzed_key=None, analyzing=False, max_reached=1, step=1)


def set_image(image_bytes: bytes, upload_sig=None) -> None:
    ss = st.session_state
    uri = image_bytes_to_data_uri(image_bytes)          # raises ValueError if unreadable
    clear_analysis()
    ss.image_bytes = image_bytes
    ss.img_uri = uri
    ss.image_key = hashlib.sha1(image_bytes).hexdigest()
    ss.upload_sig = upload_sig
    ss.detection_error = None


def go_home() -> None:
    st.session_state.screen = "home"


def start_app() -> None:
    st.session_state.screen = "app"
    st.session_state.step = 1


def go_step(n: int) -> None:
    if n <= st.session_state.max_reached:
        st.session_state.step = n


def load_sample() -> None:
    ss = st.session_state
    try:
        if not SAMPLE_IMAGE_PATH.exists():
            raise FileNotFoundError(f"Sample meal image not found: {SAMPLE_IMAGE_PATH}")
        set_image(SAMPLE_IMAGE_PATH.read_bytes(), upload_sig=None)   # same pipeline as an upload
        ss.uploader_n += 1
    except Exception as exc:
        logger.exception("Could not load sample meal")
        ss.detection_error = f"Could not load the sample meal ({type(exc).__name__}): {exc}"


def request_analysis() -> None:
    """Button callback: only flags the request. The heavy work runs in the page body
    (so a spinner can be shown), and never during Confirm / Change / Unsupported."""
    ss = st.session_state
    ss.detection_error = None
    if ss.analyzed_key and ss.analyzed_key == ss.image_key:
        ss.step = 2                      # same image already analysed: keep the user's decisions
        return
    ss.analyzing = True


def detections_from_result(result: dict, image_size: tuple[int, int],
                           labels: dict[int, str]) -> list[dict]:
    """Adapter: real pipeline output -> review-UI detection objects (variable length)."""
    width, height = image_size
    res_w, res_h = (int(v) for v in result["original_size"])
    if (res_w, res_h) != (width, height):
        raise RuntimeError(f"Inference reported image size {res_w}x{res_h} but the image is "
                           f"{width}x{height}; boxes would be misaligned.")

    parsed = []
    for info in result["boxes"]:
        class_id = int(info["foodseg103_class_id"])
        name = labels.get(class_id)
        if name is None:                 # background (0) or an id outside FoodSeg103
            continue
        x0, y0, x1, y1 = (float(v) for v in info["box"])
        parsed.append((int(info["area"]), class_id, name, (x0, y0, x1, y1)))
    parsed.sort(key=lambda p: -p[0])     # largest regions first; numbering follows this order

    dets = []
    for det_id, (area, class_id, name, (x0, y0, x1, y1)) in enumerate(parsed):
        # original-image pixels -> percentages of the ORIGINAL width/height
        left = max(0.0, min(100.0, x0 / width * 100.0))
        top = max(0.0, min(100.0, y0 / height * 100.0))
        box_w = max(0.0, min(100.0 - left, (x1 - x0 + 1) / width * 100.0))
        box_h = max(0.0, min(100.0 - top, (y1 - y0 + 1) / height * 100.0))
        dets.append({
            "id": det_id,
            "pred": name,
            "pred_class_id": class_id,
            "foodseg103_class_id": class_id,
            "food": name,
            "box": {"l": round(left, 3), "t": round(top, 3), "w": round(box_w, 3), "h": round(box_h, 3)},
            "box_px": [x0, y0, x1, y1],
            "area": area,
            "status": "needs_review",
        })
    return dets


def analyze_current_image() -> None:
    """Runs real inference. Called from the Upload page only when 'Analyze meal' was clicked."""
    ss = st.session_state
    stage = "Analysis failed"
    try:
        if not ss.image_bytes:
            raise ValueError("No meal image is available for analysis.")
        labels = load_foodseg_labels()
        stage = "Could not load the NutriVision model"
        model = get_model()                              # cached; spinner only on first load
        stage = "Analysis failed"
        with st.spinner("Analyzing your meal\u2026"):
            from src.segmentation.inference import predict
            image = decode_image(ss.image_bytes)         # ORIGINAL bytes, not the preview
            result = predict(model, np.asarray(image, dtype=np.uint8))
            dets = detections_from_result(result, image.size, labels)
    except Exception as exc:
        logger.exception("%s", stage)
        ss.detections = []
        ss.detection_error = f"{stage} ({type(exc).__name__}): {exc}"
        return

    ss.detections = dets
    ss.analyzed_key = ss.image_key
    ss.quantities = {}
    ss.qty_errors = []
    ss.max_reached = max(ss.max_reached, 2)
    ss.step = 2


def invalidate_results() -> None:
    """A review decision changed: results are stale and Quantity/Results must be re-entered
    through 'Continue to quantities'."""
    ss = st.session_state
    ss.nutrition = None
    ss.nutrition_error = None
    ss.max_reached = min(ss.max_reached, 2)


def set_status(i: int, status: str) -> None:
    ss = st.session_state
    d = ss.detections[i]
    if status == "confirmed":
        invalidate_results()
        ss.pop(f"sel_{i}", None)
        d["status"] = "confirmed"
        d["food"] = d["pred"]
        d["foodseg103_class_id"] = d["pred_class_id"]
    elif status == "unsupported":
        invalidate_results()
        ss.pop(f"sel_{i}", None)
        d["status"] = "unsupported"
        d["food"] = None
    else:
        if d["status"] in ("changing", "corrected"):
            return                        # already choosing / corrected: keep the current choice
        invalidate_results()
        d["status"] = "changing"
        d["food"] = None


def pick_food(i: int) -> None:
    d = st.session_state.detections[i]
    val = st.session_state.get(f"sel_{i}")      # a FoodSeg103 class id
    invalidate_results()
    if val is not None:
        d["status"] = "corrected"
        d["food"] = load_foodseg_labels()[int(val)]
        d["foodseg103_class_id"] = int(val)
    else:
        d["status"] = "changing"
        d["food"] = None


def get_unique_supported_foods() -> list[dict]:
    """Review is detection-level; quantity/nutrition are food-level.

    One entry per FINAL FoodSeg103 class id among confirmed/corrected detections (first-seen
    order). Unsupported detections are not FoodSeg103 classes and never appear here.
    """
    labels = load_foodseg_labels()
    seen: dict[int, dict] = {}
    for d in st.session_state.detections:
        if d["status"] in ("confirmed", "corrected") and d["food"]:
            cid = int(d["foodseg103_class_id"])
            seen.setdefault(cid, {"foodseg103_class_id": cid, "food": labels[cid]})
    return list(seen.values())


def build_qty_screen() -> None:
    ss = st.session_state
    ids = {f["foodseg103_class_id"] for f in get_unique_supported_foods()}
    ss.quantities = {cid: g for cid, g in ss.quantities.items() if cid in ids}   # drop stale foods
    ss.nutrition = None
    ss.nutrition_error = None
    ss.max_reached = max(ss.max_reached, 3)
    ss.qty_errors = []
    ss.step = 3


def remember_qty(class_id: int) -> None:
    val = st.session_state.get(f"qty_{class_id}")
    st.session_state.quantities[class_id] = val
    if class_id in st.session_state.qty_errors:
        st.session_state.qty_errors.remove(class_id)


def compute_nutrition() -> None:
    """Validate one positive quantity per unique food, then run the Stage 7 engine."""
    ss = st.session_state
    ss.nutrition = None
    ss.nutrition_error = None
    foods = get_unique_supported_foods()
    if foods:
        errors, meal = [], []
        for f in foods:
            cid = f["foodseg103_class_id"]
            grams = ss.quantities.get(cid)
            if grams is None or not grams > 0:
                errors.append(cid)
            else:
                meal.append({"foodseg103_class_id": cid, "grams": float(grams)})
        if errors:
            ss.qty_errors = errors
            return
        try:
            ss.nutrition = calculate_meal_nutrition(meal, get_nutrition_reference())
        except Exception as exc:
            logger.exception("Nutrition calculation failed")
            ss.nutrition_error = f"Could not calculate nutrition ({type(exc).__name__}): {exc}"
            return
    ss.qty_errors = []
    ss.max_reached = max(ss.max_reached, 4)
    ss.step = 4


def reset_all() -> None:
    ss = st.session_state
    n = ss.uploader_n + 1
    clear_analysis()
    ss.update(img_uri=None, image_bytes=None, image_key=None, upload_sig=None,
              detection_error=None, uploader_n=n)


# --------------------------------------------------------------------------
# Shared pieces
# --------------------------------------------------------------------------
def render_topbar() -> None:
    with st.container(key="topbar"):
        with st.container(key="row_topbar"):
            st.button("NutriVision", key="brand", on_click=go_home)
            st.button("Analyze a meal", key="top_cta", type="primary", on_click=start_app)


def render_stepbar() -> None:
    cur, reached = st.session_state.step, st.session_state.max_reached
    with st.container(key="row_steps"):
        for i, name in enumerate(STEP_NAMES, start=1):
            if i == cur:
                kind = "cur"
            elif i < cur and i <= reached:
                kind = "done"
            else:
                kind = "off"
            st.button(name, key=f"step{kind}_{i}", disabled=i > reached,
                      on_click=go_step, args=(i,))


def preview_html(uri: str, detections=None) -> str:
    """Image + overlay. Boxes are % of the ORIGINAL image; the <img> is width:100%/height:auto
    (no crop, no distortion), so the % coordinate system matches the displayed pixels exactly."""
    boxes = ""
    for i, d in enumerate(detections or []):
        status = d["status"]
        if status == "unsupported":
            cls, label = "b-bad", f"{i + 1} unsupported"
        elif status == "confirmed":
            cls, label = "b-ok", f"{i + 1} {escape(d['food'] or d['pred'])}"
        elif status == "corrected":
            cls, label = "b-warn", f"{i + 1} {escape(d['food'] or d['pred'])}"
        else:  # needs_review / changing: still showing the model's prediction
            cls, label = "b-warn", f"{i + 1} {escape(d['pred'])}"
        b = d["box"]
        if b["t"] < 7:
            cls += " lbl-in"     # keep the label inside the image when the box touches the top edge
        boxes += (f'<div class="box {cls}" style="left:{b["l"]}%;top:{b["t"]}%;'
                  f'width:{b["w"]}%;height:{b["h"]}%;"><span>{label}</span></div>')
    return f'<div class="preview"><img src="{uri}" alt="Meal photo">{boxes}</div>'


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
def render_home() -> None:
    hero_uri = file_to_data_uri(str(HERO_IMAGE_PATH))
    with st.container(key="home"):
        with st.container(key="hero"):
            left, right = st.columns([1.1, 1], gap="large", vertical_alignment="center")
            with left:
                html("""
                <div class="hero-copy">
                <div class="hero-title">See what's really on your plate.</div>
                <div class="lede">Take a photo of your meal and NutriVision identifies each food on it,
                then walks you through confirming what it found so your nutrition numbers are actually
                right \u2014 not guessed.</div>
                </div>
                """)
                with st.container(key="row_cta"):
                    st.button("Analyze your meal", key="hero_cta", type="primary", on_click=start_app)
                    html('<a class="btn-text" href="#how" target="_self">See how it works</a>')
            with right:
                if hero_uri:
                    html(f'<div class="plate-card"><img src="{hero_uri}" alt="Sample meal illustration"></div>')

        cards = "".join(
            f'<div class="how-card"><div class="n">{n}</div><div class="how-h">{h}</div>'
            f'<div class="how-p">{p}</div></div>'
            for n, h, p in HOW_STEPS
        )
        html(f'<div class="how" id="how"><div class="how-title">How NutriVision works</div>'
             f'<div class="how-grid">{cards}</div></div>')

        html("""
        <div class="honest-inner"><b>Worth knowing:</b> NutriVision identifies foods from your photo,
        but it doesn't guess portion size or calories from the image alone \u2014 you tell it how much you
        ate, and it looks up the nutrition from there. If it spots something outside its food list,
        it'll say so instead of quietly guessing.</div>
        """)


# --------------------------------------------------------------------------
# APP STEPS
# --------------------------------------------------------------------------
def render_step_upload() -> None:
    ss = st.session_state
    busy = ss.analyzing          # analysis requested by the button on the previous run
    ss.analyzing = False         # consumed: an interrupted run must never silently resume
    with st.container(key="card_upload"):
        html('<div class="st-title">Upload your meal photo</div>'
             '<div class="substep">One clear photo of the whole plate works best.</div>')

        has_image = bool(ss.img_uri)
        with st.container(key="drop_has" if has_image else "drop_empty"):
            up = st.file_uploader(
                "Meal photo", type=["jpg", "jpeg", "png"],
                key=f"uploader_{ss.uploader_n}", label_visibility="collapsed",
            )
        if up is not None:
            sig = (up.name, up.size)
            if ss.upload_sig != sig:
                try:
                    set_image(up.getvalue(), upload_sig=sig)
                except ValueError as exc:
                    clear_analysis()
                    ss.update(image_bytes=None, img_uri=None, image_key=None,
                              upload_sig=sig, detection_error=str(exc))
                st.rerun()

        with st.container(key="row_sample"):
            st.button("Or try a sample meal", key="sample_btn", type="tertiary",
                      on_click=load_sample, disabled=busy)

        if has_image:
            html(preview_html(ss.img_uri))

        if ss.detection_error:
            html(f'<div class="stopbanner">{escape(ss.detection_error)}</div>')
        with st.container(key="foot_in_card"):
            st.button("Analyze meal", key="analyze_btn", type="primary",
                      disabled=(not has_image) or busy, on_click=request_analysis)

        if busy and has_image:
            analyze_current_image()          # spinner(s) render here; blocks only this run
            if ss.detection_error:           # failed: stay on Upload and show why
                html(f'<div class="stopbanner">{escape(ss.detection_error)}</div>')
            else:
                st.rerun()                   # success: step is now 2 -> Review


def render_step_review() -> None:
    ss = st.session_state
    dets = ss.detections
    labels = load_foodseg_labels()

    with st.container(key="card_review_head"):
        html('<div class="st-title">Review what we found</div>'
             '<div class="substep">Confirm each item, fix it, or mark it unsupported. '
             'Every item needs a decision.</div>')
        if ss.img_uri:
            html(preview_html(ss.img_uri, dets))

    if not dets:
        with st.container(key="card_review_empty"):
            html('<div class="stopbanner" style="margin-bottom:0;">No supported food items were '
                 'detected. Try another meal photo.</div>')

    for i, d in enumerate(dets):
        status = d["status"]
        if status == "unsupported":
            badge_cls, badge = "bg-bad", "Unsupported"
        elif status == "corrected":
            badge_cls, badge = "bg-warn", f"Corrected \u2192 {escape(d['food'])}"
        elif status == "confirmed":
            badge_cls, badge = "bg-ok", "Confirmed"
        elif status == "changing":
            badge_cls, badge = "bg-warn", "Choose a food"
        else:
            badge_cls, badge = "bg-warn", "Needs review"
        if status in ("confirmed", "corrected") and d["food"]:
            title = escape(d["food"])
        elif status == "needs_review":
            title = escape(d["pred"])
        else:
            title = f"Item {i + 1}"

        with st.container(key=f"card_item_{i}"):
            html(f"""
            <div class="item-head"><span class="item-name">{i + 1} &nbsp;{title}</span>
            <span class="badge {badge_cls}">{badge}</span></div>
            <div class="conf">Model predicted {escape(d['pred'])} \u00b7 FoodSeg103 class {d['pred_class_id']}</div>
            """)
            with st.container(key=f"pills_{i}"):
                st.button("Confirm", key=f"pill_c_{i}",
                          type="primary" if status == "confirmed" else "secondary",
                          on_click=set_status, args=(i, "confirmed"))
                st.button("Change", key=f"pill_x_{i}",
                          type="primary" if status in ("corrected", "changing") else "secondary",
                          on_click=set_status, args=(i, "changing"))
                st.button("Unsupported", key=f"pill_u_{i}",
                          type="primary" if status == "unsupported" else "secondary",
                          on_click=set_status, args=(i, "unsupported"))
            if status in ("changing", "corrected"):
                # real FoodSeg103 vocabulary only: no background, no "Unsupported", not the current prediction
                # options = sorted((cid for cid in labels if cid != d["pred_class_id"]),
                #                  key=lambda cid: labels[cid].lower())
                options = sorted(labels.keys(), key=lambda cid: labels[cid].lower())
                index = options.index(d["foodseg103_class_id"]) if status == "corrected" else None
                st.selectbox("Supported food", options, index=index, key=f"sel_{i}",
                             format_func=lambda cid: labels[cid],
                             placeholder="Select a supported food\u2026",
                             label_visibility="collapsed", on_change=pick_food, args=(i,))

    all_resolved = bool(dets) and all(
        d["status"] in ("confirmed", "unsupported") or (d["status"] == "corrected" and d["food"])
        for d in dets
    )
    with st.container(key="foot_lr"):
        st.button("Back", key="review_back", on_click=go_step, args=(1,))
        st.button("Continue to quantities", key="to_qty", type="primary",
                  disabled=not all_resolved, on_click=build_qty_screen)


def render_step_quantity() -> None:
    ss = st.session_state
    supported = get_unique_supported_foods()

    with st.container(key="card_qty"):
        html('<div class="st-title">How much did you eat?</div>'
             '<div class="note">NutriVision doesn\'t estimate portion size from the photo \u2014 '
             'enter the amount in grams for each confirmed food.</div>')
        if not supported:
            html('<div class="stopbanner">No supported foods were confirmed, so there\'s nothing to '
                 'weigh yet. Go back and confirm or correct at least one item.</div>')
        for f in supported:
            cid = f["foodseg103_class_id"]
            with st.container(key=f"qty_item_{cid}"):
                html(f'<div class="qtyname">{escape(f["food"])}</div>')
                st.number_input(
                    f"Grams of {f['food']}", min_value=1.0, step=1.0, format="%g",
                    value=ss.quantities.get(cid), placeholder="Grams",
                    key=f"qty_{cid}", label_visibility="collapsed",
                    on_change=remember_qty, args=(cid,),
                )
                if cid in ss.qty_errors:
                    html('<div class="err">Enter a quantity greater than zero.</div>')
        if ss.nutrition_error:
            html(f'<div class="stopbanner">{escape(ss.nutrition_error)}</div>')

    with st.container(key="foot_lr"):
        st.button("Back", key="qty_back", on_click=go_step, args=(2,))
        st.button("Calculate nutrition", key="to_nut", type="primary", on_click=compute_nutrition)


def render_step_results() -> None:
    ss = st.session_state
    foods = get_unique_supported_foods()
    excluded = sum(1 for d in ss.detections if d["status"] == "unsupported")
    meal = ss.nutrition
    labels = load_foodseg_labels()

    with st.container(key="card_results"):
        if not foods:
            plural = "s were" if excluded > 1 else " was"
            html(f"""
            <div class="st-title">No nutrition to show</div>
            <div class="stopbanner" style="margin-top:10px;">All {excluded} detected item{plural}
            unsupported, so NutriVision can't calculate nutrition for this meal. Go back and review the
            items, or start a new meal.</div>
            """)
        elif meal is None:
            html("""
            <div class="st-title">No nutrition to show</div>
            <div class="stopbanner" style="margin-top:10px;">Nutrition hasn't been calculated yet.
            Go back to Quantity and press Calculate nutrition.</div>
            """)
        else:
            n = len(foods)
            sub = f"{n} food{'s' if n > 1 else ''} included"
            if excluded:
                sub += f" \u00b7 {excluded} unsupported item{'s' if excluded > 1 else ''} excluded"
            banner = ""
            if excluded:
                banner = (f'<div class="banner">{excluded} detected item{"s were" if excluded > 1 else " was"} '
                          f'unsupported and left out of the totals below.</div>')
            def fmt(v, nd=1, unit=""):
                return "\u2014" if v is None else f"{v:.{nd}f}{unit}"

            totals = meal["totals"]
            metrics = "".join(
                f'<div class="metric"><b>{fmt(totals[key], nd, unit)}</b><span>{lab}</span></div>'
                for lab, key, nd, unit in (
                    ("Calories", "calories_kcal", 0, " kcal"), ("Protein", "protein_g", 1, " g"),
                    ("Carbs", "carb_g", 1, " g"), ("Fat", "fat_total_g", 1, " g"),
                    ("Fiber", "fiber_g", 1, " g"))
            )
            rows = ""
            for r in meal["foods"]:
                name = labels.get(r["foodseg103_class_id"], r["class_name"])
                rows += (f"<tr><td>{escape(name)}</td><td>{r['grams']:g} g</td>"
                         f"<td>{fmt(r['calories_kcal'], 0)}</td><td>{fmt(r['protein_g'])}</td>"
                         f"<td>{fmt(r['carb_g'])}</td><td>{fmt(r['fat_total_g'])}</td>"
                         f"<td>{fmt(r['fiber_g'])}</td></tr>")
            rows += (f"<tr><td><b>Meal total</b></td><td>{fmt(sum(r['grams'] for r in meal['foods']), 0, ' g')}</td>"
                     f"<td><b>{fmt(totals['calories_kcal'], 0)}</b></td><td><b>{fmt(totals['protein_g'])}</b></td>"
                     f"<td><b>{fmt(totals['carb_g'])}</b></td><td><b>{fmt(totals['fat_total_g'])}</b></td>"
                     f"<td><b>{fmt(totals['fiber_g'])}</b></td></tr>")
            html(f"""
            <div class="st-title">Meal nutrition</div>
            <div class="substep">{sub}</div>
            {banner}
            <div class="metrics">{metrics}</div>
            <table><tr><th>Food</th><th>Qty</th><th>Cal</th><th>Protein</th><th>Carbs</th><th>Fat</th><th>Fiber</th></tr>{rows}</table>
            <div class="note" style="margin-top:12px;">Based on the quantities you entered \u2014
            NutriVision doesn't estimate weight from the photo.</div>
            """)

    with st.container(key="foot_l"):
        st.button("Start a new meal", key="reset_btn", on_click=reset_all)


def render_app() -> None:
    with st.container(key="app"):
        render_stepbar()
        with st.container(key="wrap"):
            {1: render_step_upload, 2: render_step_review,
             3: render_step_quantity, 4: render_step_results}[st.session_state.step]()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    init_state()
    load_css()
    render_topbar()
    if st.session_state.screen == "home":
        render_home()
    else:
        render_app()


main()