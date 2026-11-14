"""Composition Critic API: everything in one file.

Run:
    pip install fastapi uvicorn python-multipart anthropic
    uvicorn app:app --reload --port 8000

With no ANTHROPIC_API_KEY set, the LLM part returns fixed demo data,
so this runs without a key. Set the key and restart to use the model.

Sections:
    1. LLM: labels and explanations (C)
    2. Attention: eye path and focal scores (A), currently a STUB
    3. Metrics: design scores (B), currently a STUB
    4. API: total, band and endpoints
"""

import base64
import json
import os
import tempfile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware


# =====================================================================
# 1. LLM: labels and explanations (C)
#    The LLM never scores. It names what is in the image and turns the
#    measured numbers into plain advice.
# =====================================================================

MODEL = os.environ.get("CRITIC_MODEL", "claude-sonnet-5-5")

MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}

ROLES = ["headline", "offer", "logo", "product", "person", "body_text", "call_to_action", "background", "decoration", "other"]


def _use_stub() -> bool:
    return not os.environ.get("ANTHROPIC_API_KEY")


def _image_block(path: str) -> dict:
    suffix = os.path.splitext(path)[1].lower()
    media_type = MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise ValueError(f"Unsupported image type {suffix!r}: use JPEG, PNG or WebP")
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _call_tool(system: str, content: list, tool: dict) -> dict:
    """One model call that must answer through `tool`; returns the tool's input."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        temperature=0,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return block.input
    raise ValueError("Model did not return structured output")


# ------------------------------------------------------------ 1. labels

LABEL_SYSTEM = """You label the visual elements of a small-business social media post or advert.
List every element a viewer could notice: headline, offer, logo, product photo, people, body text, call to action, notable decoration.
For each, give a bounding box as fractions of the image width and height (0 to 1), measured from the top-left corner.
Mark is_objective true for the element or elements that carry the owner's stated objective, and false for the rest.
Use short, concrete names a shop owner would understand, such as "50% OFF text" or "shop logo"."""

LABEL_TOOL = {
    "name": "submit_elements",
    "description": "Submit the labelled elements.",
    "input_schema": {
        "type": "object",
        "properties": {
            "elements": {
                "type": "array",
                "minItems": 1,
                "maxItems": 12,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {"type": "string", "enum": ROLES},
                        "box": {
                            "type": "object",
                            "properties": {k: {"type": "number", "minimum": 0, "maximum": 1} for k in ("x", "y", "w", "h")},
                            "required": ["x", "y", "w", "h"],
                        },
                        "is_objective": {"type": "boolean"},
                    },
                    "required": ["name", "role", "box", "is_objective"],
                },
            }
        },
        "required": ["elements"],
    },
}

STUB_ELEMENTS = [
    {"name": "Shop logo", "role": "logo", "box": {"x": 0.30, "y": 0.05, "w": 0.40, "h": 0.25}, "is_objective": False},
    {"name": "50% OFF text", "role": "offer", "box": {"x": 0.60, "y": 0.70, "w": 0.30, "h": 0.12}, "is_objective": True},
    {"name": "Product photo", "role": "product", "box": {"x": 0.10, "y": 0.35, "w": 0.45, "h": 0.40}, "is_objective": False},
    {"name": "Shop now button", "role": "call_to_action", "box": {"x": 0.35, "y": 0.88, "w": 0.30, "h": 0.07}, "is_objective": False},
]


def label_elements(path: str, objective: str) -> list[dict]:
    """Return the elements in the image, each with id, name, role, box (0-1 fractions) and is_objective."""
    if _use_stub():
        raw = [dict(e) for e in STUB_ELEMENTS]
    else:
        result = _call_tool(
            LABEL_SYSTEM,
            [_image_block(path), {"type": "text", "text": f"The owner's objective (what the audience should notice first):\n{objective}"}],
            LABEL_TOOL,
        )
        raw = result["elements"]

    elements = []
    for i, e in enumerate(raw, start=1):
        box = {k: min(max(float(e["box"][k]), 0.0), 1.0) for k in ("x", "y", "w", "h")}
        elements.append({"id": f"e{i}", "name": e["name"], "role": e["role"], "box": box, "is_objective": bool(e["is_objective"])})
    if not any(e["is_objective"] for e in elements):
        raise ValueError("Could not find the objective in the image: try rewording it")
    return elements


# ------------------------------------------------------------ 2. explanations

EXPLAIN_SYSTEM = """You are a composition coach for small-business owners. You do not score; the scores are already measured.
You get the owner's objective, the predicted eye path (which elements the viewer looks at, in order) and subscores.
Explain why the design does or does not get the objective noticed first, based on those numbers.

Rules:
- Problems: up to 3, worst first. Each names a specific element and says what it does to the viewer's attention, in plain words with no design jargon.
- Fixes: exactly 3, biggest impact first. Each names one element and one concrete change (move, resize, recolour or remove, with a rough amount, such as "about half the size").
- Link each fix to the subscore it should raise, using the subscore's exact key."""


def _explain_tool(subscore_keys: list[str]) -> dict:
    return {
        "name": "submit_explanation",
        "description": "Submit problems and fixes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "problems": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {"element": {"type": "string"}, "explanation": {"type": "string"}},
                        "required": ["element", "explanation"],
                    },
                },
                "fixes": {
                    "type": "array",
                    "minItems": 3,
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "element": {"type": "string"},
                            "change": {"type": "string"},
                            "improves": {"type": "string", "enum": subscore_keys},
                        },
                        "required": ["element", "change", "improves"],
                    },
                },
            },
            "required": ["problems", "fixes"],
        },
    }


def explain(path: str, objective: str, eye_path: list[dict], subscores: dict) -> tuple[list, list]:
    """Return (problems, fixes) in plain language, based on the measured eye path and subscores."""
    if _use_stub():
        problems = [
            {"element": "Shop logo", "explanation": "The logo is the biggest, brightest thing in the post, so people look there first and may scroll on before reaching the discount."},
            {"element": "50% OFF text", "explanation": "The discount sits in the bottom corner in small text, so it is only the third thing people see."},
        ]
        fixes = [
            {"element": "50% OFF text", "change": "Make it about three times larger and move it to the centre of the post.", "improves": "objective_first"},
            {"element": "Shop logo", "change": "Shrink it to about a third of its size and move it to a top corner.", "improves": "attention_share"},
            {"element": "50% OFF text", "change": "Put it on a solid block of a colour that does not appear elsewhere in the post.", "improves": "contrast"},
        ]
        return problems, fixes

    path_summary = [
        {"stop": p.get("order"), "element": p.get("element_name") or "empty area"} for p in eye_path
    ]
    context = {"objective": objective, "eye_path": path_summary, "subscores": subscores}
    result = _call_tool(
        EXPLAIN_SYSTEM,
        [_image_block(path), {"type": "text", "text": json.dumps(context, indent=2)}],
        _explain_tool(list(subscores)),
    )
    return result["problems"], result["fixes"][:3]


# =====================================================================
# 2. Attention: eye path and focal scores (A)
#    STUB: fixed numbers. Replace the bodies of get_eye_path and
#    focal_scores, keeping the same names, arguments and return shapes.
#    Worth 40 of the 100 points:
#      objective_first  0-20  how early the eye reaches the objective
#      attention_share  0-20  how much attention the objective gets
# =====================================================================

FOCAL_MAX = {"objective_first": 20, "attention_share": 20}


def get_eye_path(path: str) -> list[dict]:
    """Return predicted fixations in viewing order.

    Each fixation: {"order": 1.., "x": 0-1, "y": 0-1} as fractions of
    image width/height from the top-left corner.
    """
    return [
        {"order": 1, "x": 0.50, "y": 0.17},
        {"order": 2, "x": 0.32, "y": 0.55},
        {"order": 3, "x": 0.75, "y": 0.76},
        {"order": 4, "x": 0.50, "y": 0.91},
    ]


def label_path(eye_path: list[dict], elements: list[dict]) -> list[dict]:
    """Attach the element under each fixation (smallest box containing the point).

    Adds "element_id" and "element_name" to each fixation, or None if the
    fixation lands on no element. This part is real, not stubbed, so it
    works with the LLM's labels straight away.
    """
    labelled = []
    for fix in eye_path:
        hits = [
            e for e in elements
            if e["box"]["x"] <= fix["x"] <= e["box"]["x"] + e["box"]["w"]
            and e["box"]["y"] <= fix["y"] <= e["box"]["y"] + e["box"]["h"]
        ]
        hit = min(hits, key=lambda e: e["box"]["w"] * e["box"]["h"]) if hits else None
        labelled.append({**fix, "element_id": hit["id"] if hit else None, "element_name": hit["name"] if hit else None})
    return labelled


def focal_scores(eye_path: list[dict], elements: list[dict], objective: str) -> dict:
    """STUB: fixed scores. Must return exactly the keys in FOCAL_MAX, each within 0..max."""
    return {"objective_first": 8, "attention_share": 10}


# =====================================================================
# 3. Metrics: design scores (B)
#    STUB: fixed numbers. Replace the body of design_scores, keeping the
#    same name, arguments and return shape.
#    Worth 60 of the 100 points:
#      contrast    0-20  does the objective stand out from its surroundings
#      clutter     0-20  how few competing elements there are (20 = clean)
#      legibility  0-20  is the key text readable at phone size
# =====================================================================

DESIGN_MAX = {"contrast": 20, "clutter": 20, "legibility": 20}


def design_scores(path: str, elements: list[dict]) -> dict:
    """STUB: fixed scores. Must return exactly the keys in DESIGN_MAX, each within 0..max."""
    return {"contrast": 12, "clutter": 14, "legibility": 16}


# =====================================================================
# 4. API: total, band and endpoints
# =====================================================================

app = FastAPI()

# Lets the frontend (D) call this server from the browser
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

BANDS = [(85, "Ready to launch"), (65, "Almost there"), (40, "Needs work"), (0, "Will be ignored")]

# Every subscore and its maximum. Together they must add up to 100.
MAX_POINTS = {**FOCAL_MAX, **DESIGN_MAX}
assert not set(FOCAL_MAX) & set(DESIGN_MAX), "A and B use the same subscore name"
assert sum(MAX_POINTS.values()) == 100, f"Subscore maximums add up to {sum(MAX_POINTS.values())}, not 100"

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def check_scores(scores: dict, maxima: dict, owner: str) -> dict:
    """Make sure a teammate's function returned exactly the agreed keys, in range."""
    if set(scores) != set(maxima):
        raise ValueError(f"{owner} returned keys {sorted(scores)}, expected {sorted(maxima)}")
    return {k: min(max(float(v), 0.0), maxima[k]) for k, v in scores.items()}


def band_for(total: int) -> str:
    return next(label for minimum, label in BANDS if total >= minimum)


@app.post("/analyse")
async def analyse(image: UploadFile = File(...), objective: str = Form(...)):
    objective = objective.strip()
    if not objective:
        raise HTTPException(422, "Tell us what the audience should notice first")
    suffix = os.path.splitext(image.filename or "")[1].lower() or ".png"
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(415, "Upload a JPEG, PNG or WebP image")

    # Save the upload to a temp file so everyone's functions can read it by path
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        path = tmp.name

    step = "starting"
    try:
        step = "labelling elements (C)"
        elements = label_elements(path, objective)
        step = "predicting eye path (A)"
        eye_path = get_eye_path(path)
        eye_path = label_path(eye_path, elements)
        step = "scoring attention (A)"
        focal = check_scores(focal_scores(eye_path, elements, objective), FOCAL_MAX, "focal_scores")
        step = "scoring design (B)"
        design = check_scores(design_scores(path, elements), DESIGN_MAX, "design_scores")

        subscores = {**focal, **design}
        total = int(round(sum(subscores.values())))
        band = band_for(total)

        step = "writing explanations (C)"
        problems, fixes = explain(path, objective, eye_path, subscores)

        return {
            "objective": objective, "elements": elements, "eye_path": eye_path,
            "subscores": subscores, "max_points": MAX_POINTS,
            "total": total, "band": band,
            "problems": problems, "fixes": fixes,
        }
    except Exception as exc:
        raise HTTPException(500, f"Failed while {step}: {exc}")
    finally:
        os.remove(path)


@app.get("/rubric")
def rubric():
    """For the frontend: what each subscore is worth and where the bands start."""
    return {"max_points": MAX_POINTS, "bands": [{"min_total": m, "band": b} for m, b in BANDS]}


@app.get("/")
def health():
    return {"status": "ok", "llm": "stub" if _use_stub() else MODEL}