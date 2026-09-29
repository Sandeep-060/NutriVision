from __future__ import annotations
import base64
import io
from html import escape
from pathlib import Path
import streamlit as st
from PIL import Image, ImageOps

APP_DIR = Path(__file__).resolve().parent
HERO_IMAGE_PATH = APP_DIR / "assets" / "hero_image.png"
CSS_PATH = APP_DIR / "style.css"

# --------------------------------------------------------------------------
# Static data (mock). Labels mirror the prototype's supported food list.
# --------------------------------------------------------------------------
LABELS = {
    "rice": "Rice", "chapati": "Chapati", "dal": "Dal", "sambar": "Sambar",
    "rasam": "Rasam", "curd_raita": "Curd / raita", "paneer_curry": "Paneer curry",
    "potato_curry": "Potato curry", "chicken_curry": "Chicken curry",
    "biryani": "Biryani", "idli": "Idli", "dosa": "Dosa", "vada": "Vada",
    "poori": "Poori", "samosa": "Samosa", "naan": "Naan", "papad": "Papad",
}
LABEL_TO_KEY = {v: k for k, v in LABELS.items()}

MOCK_DETECTIONS = [
    {"id": 0, "pred": "rice", "conf": 0.94, "box": {"l": 5, "t": 8, "w": 42, "h": 36}},
    {"id": 1, "pred": "sambar", "conf": 0.58, "box": {"l": 55, "t": 5, "w": 34, "h": 34}},
    {"id": 2, "pred": "chapati", "conf": 0.89, "box": {"l": 9, "t": 51, "w": 36, "h": 36}},
    {"id": 3, "pred": "paneer_curry", "conf": 0.41, "box": {"l": 53, "t": 47, "w": 38, "h": 38}},
]
THRESHOLD = 0.65

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

st.set_page_config(
    page_title="NutriVision",
    page_icon="\U0001F957",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_css() -> None:
    css = CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render a small presentation snippet.

    Lines are stripped and joined so Markdown never treats indented HTML as a
    code block (which is what makes raw tags show up on screen).
    """
    flat = " ".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def svg_data_uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode("utf-8")).decode("ascii")


def plate_svg(bg: str) -> str:
    """The prototype's sample-meal illustration (same shapes and colours)."""
    return f"""<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid slice">
<rect x="0" y="0" width="100" height="100" fill="{bg}"/>
<ellipse cx="26" cy="26" rx="21" ry="18" fill="#F3E9CB"/>
<ellipse cx="20" cy="21" rx="2.1" ry="1.3" fill="#E4D6A8"/><ellipse cx="30" cy="18" rx="2" ry="1.2" fill="#E4D6A8"/>
<ellipse cx="33" cy="30" rx="2" ry="1.2" fill="#E4D6A8"/><ellipse cx="18" cy="32" rx="2" ry="1.2" fill="#E4D6A8"/>
<circle cx="72" cy="22" r="17" fill="#E7DFC9"/><circle cx="72" cy="22" r="13.5" fill="#C9752E"/>
<circle cx="67" cy="18" r="1.2" fill="#8FBE6E"/><circle cx="76" cy="26" r="1.1" fill="#8FBE6E"/><circle cx="72" cy="15" r="1" fill="#8FBE6E"/>
<circle cx="27" cy="69" r="18" fill="#D9A360"/>
<path d="M12,69 Q27,60 42,69" stroke="#B87F3E" stroke-width="1" fill="none" opacity=".6"/>
<path d="M12,73 Q27,66 42,73" stroke="#B87F3E" stroke-width="1" fill="none" opacity=".6"/>
<path d="M14,78 Q27,72 40,78" stroke="#B87F3E" stroke-width="1" fill="none" opacity=".6"/>
<circle cx="72" cy="66" r="19" fill="#E7DFC9"/><circle cx="72" cy="66" r="15.5" fill="#A63D22"/>
<rect x="65" y="60" width="7" height="7" rx="1.3" fill="#FBF6EA"/><rect x="74" y="68" width="6.5" height="6.5" rx="1.3" fill="#FBF6EA"/>
<rect x="68" y="70" width="6" height="6" rx="1.2" fill="#FBF6EA"/></svg>"""


HERO_IMAGE_URI = (
    "data:image/png;base64,"
    + base64.b64encode(HERO_IMAGE_PATH.read_bytes()).decode("ascii")
)
SAMPLE_IMAGE_URI = svg_data_uri(plate_svg("#EFEAD9"))  # prototype's sample photo


def image_to_data_uri(uploaded) -> str:
    """Downscale an uploaded photo so the preview HTML stays small."""
    img = Image.open(io.BytesIO(uploaded.getvalue()))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((900, 900))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------
# Session state + callbacks
# --------------------------------------------------------------------------
def init_state() -> None:
    defaults = {
        "screen": "home",       # "home" | "app"
        "step": 1,              # 1..4 (inside the app)
        "max_reached": 1,
        "img_uri": None,
        "upload_sig": None,
        "detections": [],
        "quantities": {},
        "qty_errors": [],
        "uploader_n": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def go_home() -> None:
    st.session_state.screen = "home"


def start_app() -> None:
    st.session_state.screen = "app"
    st.session_state.step = 1


def go_step(n: int) -> None:
    if n <= st.session_state.max_reached:
        st.session_state.step = n


def load_sample() -> None:
    st.session_state.img_uri = SAMPLE_IMAGE_URI
    st.session_state.upload_sig = None
    st.session_state.uploader_n += 1  # clears the file_uploader


def run_detection() -> None:
    """MOCK detection - replaced by the real YOLO pipeline later."""
    dets = []
    for d in MOCK_DETECTIONS:
        status = "confirmed" if d["conf"] >= THRESHOLD else "unsupported"
        dets.append({**d, "status": status, "food": d["pred"] if status == "confirmed" else None})
    st.session_state.detections = dets
    st.session_state.quantities = {}
    st.session_state.qty_errors = []
    st.session_state.max_reached = max(st.session_state.max_reached, 2)
    st.session_state.step = 2


def set_status(i: int, status: str) -> None:
    d = st.session_state.detections[i]
    if status == "confirmed":
        d["status"], d["food"] = "confirmed", d["pred"]
    elif status == "unsupported":
        d["status"], d["food"] = "unsupported", None
    else:
        d["status"] = "changing"


def pick_food(i: int) -> None:
    d = st.session_state.detections[i]
    val = st.session_state.get(f"sel_{i}")
    if val:
        d["status"], d["food"] = "corrected", LABEL_TO_KEY[val]
    else:
        d["status"], d["food"] = "changing", None


def build_qty_screen() -> None:
    st.session_state.max_reached = max(st.session_state.max_reached, 3)
    st.session_state.qty_errors = []
    st.session_state.step = 3


def remember_qty(det_id: int) -> None:
    val = st.session_state.get(f"qty_{det_id}")
    st.session_state.quantities[det_id] = val
    if det_id in st.session_state.qty_errors:
        st.session_state.qty_errors.remove(det_id)


def compute_nutrition() -> None:
    """PLACEHOLDER - validates quantities only; no nutrition maths yet."""
    supported = [d for d in st.session_state.detections if d["status"] != "unsupported"]
    if supported:
        errors = []
        for d in supported:
            v = st.session_state.quantities.get(d["id"])
            if not v or v <= 0:
                errors.append(d["id"])
        if errors:
            st.session_state.qty_errors = errors
            return
    st.session_state.max_reached = max(st.session_state.max_reached, 4)
    st.session_state.step = 4


def reset_all() -> None:
    n = st.session_state.uploader_n + 1
    for key in list(st.session_state.keys()):
        if key.startswith(("sel_", "qty_")):
            del st.session_state[key]
    st.session_state.update(
        step=1, max_reached=1, img_uri=None, upload_sig=None,
        detections=[], quantities={}, qty_errors=[], uploader_n=n,
    )


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
    boxes = ""
    for i, d in enumerate(detections or []):
        if d["status"] == "unsupported":
            cls, label = "b-bad", f"{i + 1} unsupported"
        elif d["status"] == "corrected":
            cls, label = "b-warn", f"{i + 1} {escape(LABELS[d['food']])}"
        else:
            cls, label = "b-ok", f"{i + 1} {escape(LABELS[d['food']] if d['food'] else LABELS[d['pred']])}"
        b = d["box"]
        boxes += (f'<div class="box {cls}" style="left:{b["l"]}%;top:{b["t"]}%;'
                  f'width:{b["w"]}%;height:{b["h"]}%;"><span>{label}</span></div>')
    return f'<div class="preview"><img src="{uri}" alt="Meal photo">{boxes}</div>'


# --------------------------------------------------------------------------
# HOME
# --------------------------------------------------------------------------
def render_home() -> None:
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
                html(f'<div class="plate-card"><img src="{HERO_IMAGE_URI}" alt="Sample meal illustration"></div>')

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
                ss.img_uri = image_to_data_uri(up)
                ss.upload_sig = sig
                st.rerun()

        with st.container(key="row_sample"):
            st.button("Or try a sample meal", key="sample_btn", type="tertiary", on_click=load_sample)

        if has_image:
            html(preview_html(ss.img_uri))

        with st.container(key="foot_in_card"):
            st.button("Analyze meal", key="analyze_btn", type="primary",
                      disabled=not has_image, on_click=run_detection)


def render_step_review() -> None:
    ss = st.session_state
    dets = ss.detections

    with st.container(key="card_review_head"):
        html('<div class="st-title">Review what we found</div>'
             '<div class="substep">Confirm each item, fix it, or mark it unsupported. '
             'Every item needs a decision.</div>')
        html(preview_html(ss.img_uri or SAMPLE_IMAGE_URI, dets))

    for i, d in enumerate(dets):
        status = d["status"]
        if status == "unsupported":
            badge_cls, badge = "bg-bad", "Unsupported"
        elif status == "corrected":
            badge_cls, badge = "bg-warn", f"Corrected \u2192 {escape(LABELS[d['food']])}"
        else:
            badge_cls, badge = "bg-ok", "Confirmed"
        title = escape(LABELS[d["food"]]) if status in ("confirmed", "corrected") and d["food"] else f"Item {i + 1}"
        below = " \u00b7 below review threshold" if d["conf"] < THRESHOLD else ""

        with st.container(key=f"card_item_{i}"):
            html(f"""
            <div class="item-head"><span class="item-name">{i + 1} &nbsp;{title}</span>
            <span class="badge {badge_cls}">{badge}</span></div>
            <div class="conf">Model predicted {escape(LABELS[d['pred']])} \u00b7 {round(d['conf'] * 100)}% confidence{below}</div>
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
                options = [LABELS[k] for k in LABELS if k != d["pred"]]
                index = options.index(LABELS[d["food"]]) if status == "corrected" and d["food"] else None
                st.selectbox("Supported food", options, index=index, key=f"sel_{i}",
                             placeholder="Select a supported food\u2026",
                             label_visibility="collapsed", on_change=pick_food, args=(i,))

    all_resolved = all(
        d["status"] in ("confirmed", "unsupported") or (d["status"] == "corrected" and d["food"])
        for d in dets
    )
    with st.container(key="foot_lr"):
        st.button("Back", key="review_back", on_click=go_step, args=(1,))
        st.button("Continue to quantities", key="to_qty", type="primary",
                  disabled=not all_resolved, on_click=build_qty_screen)


def render_step_quantity() -> None:
    ss = st.session_state
    supported = [d for d in ss.detections if d["status"] != "unsupported"]

    with st.container(key="card_qty"):
        html('<div class="st-title">How much did you eat?</div>'
             '<div class="note">NutriVision doesn\'t estimate portion size from the photo \u2014 '
             'enter the amount in grams for each confirmed food.</div>')
        if not supported:
            html('<div class="stopbanner">No supported foods were confirmed, so there\'s nothing to '
                 'weigh yet. Go back and confirm or correct at least one item.</div>')
        for d in supported:
            with st.container(key=f"qty_item_{d['id']}"):
                html(f'<div class="qtyname">{escape(LABELS[d["food"]])}</div>')
                st.number_input(
                    f"Grams of {LABELS[d['food']]}", min_value=1.0, step=1.0, format="%g",
                    value=ss.quantities.get(d["id"]), placeholder="Grams",
                    key=f"qty_{d['id']}", label_visibility="collapsed",
                    on_change=remember_qty, args=(d["id"],),
                )
                if d["id"] in ss.qty_errors:
                    html('<div class="err">Enter a quantity greater than zero.</div>')

    with st.container(key="foot_lr"):
        st.button("Back", key="qty_back", on_click=go_step, args=(2,))
        st.button("Calculate nutrition", key="to_nut", type="primary", on_click=compute_nutrition)


def render_step_results() -> None:
    ss = st.session_state
    supported = [d for d in ss.detections if d["status"] != "unsupported"]
    excluded = len(ss.detections) - len(supported)

    with st.container(key="card_results"):
        if not supported:
            plural = "s were" if excluded > 1 else " was"
            html(f"""
            <div class="st-title">No nutrition to show</div>
            <div class="stopbanner" style="margin-top:10px;">All {excluded} detected item{plural}
            unsupported, so NutriVision can't calculate nutrition for this meal. Go back and review the
            items, or start a new meal.</div>
            """)
        else:
            n = len(supported)
            sub = f"{n} supported item{'s' if n > 1 else ''} included"
            if excluded:
                sub += f" \u00b7 {excluded} unsupported item{'s' if excluded > 1 else ''} excluded"
            banner = ""
            if excluded:
                banner = (f'<div class="banner">{excluded} detected item{"s were" if excluded > 1 else " was"} '
                          f'unsupported and left out of the totals below.</div>')
            metrics = "".join(
                f'<div class="metric"><b>\u2014</b><span>{lab}</span></div>'
                for lab in ("Calories", "Protein", "Carbs", "Fat", "Fiber")
            )
            rows = ""
            for d in supported:
                grams = ss.quantities.get(d["id"])
                qty = f"{grams:g} g" if grams else "\u2014"
                rows += (f"<tr><td>{escape(LABELS[d['food']])}</td><td>{qty}</td>"
                         + "<td>\u2014</td>" * 5 + "</tr>")
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