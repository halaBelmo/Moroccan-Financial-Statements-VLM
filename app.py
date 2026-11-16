"""Streamlit upload interface for the extraction used in Local_extraction.ipynb."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

import streamlit as st
from PIL import Image

from notebook_config import MODEL_ID, notebook_settings
from notebook_pipeline import infer_group, load_model, prepare_page, render_pdf


@st.cache_resource(show_spinner="Loading Qwen2.5-VL-7B-Instruct…")
def load_model_cached():
    return load_model()

st.set_page_config(page_title="Moroccan Financial Extractor", page_icon="📄", layout="wide")
st.markdown("""
<style>
    .block-container {max-width: 1150px; padding-top: 2.2rem}
    h1 {letter-spacing: -.04em}
    .quiet {color: #64748b; font-size: .95rem}
    .stButton > button[kind="primary"] {border-radius: 8px; font-weight: 650}
</style>
""", unsafe_allow_html=True)

settings = notebook_settings()
st.title("Moroccan Financial Extractor")
st.markdown(
    f"<p class='quiet'>Upload a gridded balance sheet and extract ACTIF or PASSIF with "
    f"<b>{MODEL_ID}</b> and the prompts in <code>Local_extraction.ipynb</code>.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Notebook configuration")
    st.write("**Model:** Qwen2.5-VL-7B-Instruct, 4-bit local inference")
    st.write("**ACTIF:** prompt V7 · 14 groups · 51 extracted rows (source rows 3–53)")
    st.write("**PASSIF:** prompt V13 · 14 groups · 44 rows")
    st.info("Inference needs a CUDA GPU. The first run downloads the model weights from Hugging Face.")
    with st.expander("View extraction prompt"):
        prompt_kind = st.selectbox("Prompt", ["ACTIF", "PASSIF"])
        example_name, example_count = next(iter(settings[prompt_kind]["groups"].values()))
        st.code(settings[prompt_kind]["prompt"](example_name, example_count), language="text")

uploaded = st.file_uploader("Upload a PDF or one prepared group image", type=["pdf", "png", "jpg", "jpeg"])
if uploaded is None:
    st.caption("For PDFs, assign each page as ACTIF, PASSIF, or Skip. For an image, select its notebook group.")
    st.stop()

file_bytes = uploaded.getvalue()
if len(file_bytes) > 25 * 1024 * 1024:
    st.error("The upload limit is 25 MB.")
    st.stop()
file_hash = hashlib.sha256(file_bytes).hexdigest()
is_pdf = uploaded.name.lower().endswith(".pdf")
upload_dir = Path(__file__).resolve().parent / "local_uploads" / "files"
upload_dir.mkdir(parents=True, exist_ok=True)
safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(uploaded.name).name).strip("._")
saved_upload = upload_dir / f"{file_hash[:16]}_{safe_name or 'upload'}"
if not saved_upload.exists():
    saved_upload.write_bytes(file_bytes)
st.caption(f"Saved locally: {saved_upload.relative_to(Path(__file__).resolve().parent)}")

try:
    if is_pdf:
        pages = render_pdf(file_bytes)
        st.subheader("Pages")
        choices = []
        for i, page in enumerate(pages):
            cols = st.columns([1, 2])
            with cols[0]:
                st.image(page, caption=f"Page {i + 1}", width=180)
            with cols[1]:
                default = "ACTIF" if i == 0 else "PASSIF" if i == 1 else "Skip"
                options = [default] + [x for x in ["ACTIF", "PASSIF", "Skip"] if x != default]
                choices.append(st.selectbox(f"Page {i + 1} type", options, key=f"{file_hash}_{i}"))
        signature = (file_hash, tuple(choices))
        if st.button("Prepare group images", type="primary"):
            prepared = []
            errors = []
            for i, (page, kind) in enumerate(zip(pages, choices), 1):
                if kind == "Skip":
                    continue
                try:
                    result = prepare_page(page, kind)
                    result["page"] = i
                    prepared.append(result)
                except ValueError as exc:
                    errors.append(f"Page {i}: {exc}")
            st.session_state["prepared"] = prepared if not errors else None
            st.session_state["prepared_signature"] = signature
            st.session_state["prepare_errors"] = errors
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        kind = st.selectbox("Document type", ["ACTIF", "PASSIF"])
        options = settings[kind]["groups"]
        group_id = st.selectbox("Notebook group", list(options), format_func=lambda key: options[key][0])
        name, count = options[group_id]
        st.image(image, caption=f"{name} · {count} expected rows", width="stretch")
        signature = (file_hash, kind, group_id)
        if st.button("Use this group image", type="primary"):
            st.session_state["prepared"] = [{"kind": kind, "rows": count, "page": None,
                                             "groups": [{"id": group_id, "name": name,
                                                         "expected_rows": count, "image": image}]}]
            st.session_state["prepared_signature"] = signature
            st.session_state["prepare_errors"] = []
except Exception as exc:
    st.error(f"Could not read the uploaded file: {exc}")
    st.stop()

if st.session_state.get("prepared_signature") != signature:
    st.info("Prepare the current file and selection to continue.")
    st.stop()
for error in st.session_state.get("prepare_errors", []):
    st.error(error)
prepared = st.session_state.get("prepared")
if not prepared:
    st.stop()

st.subheader("Prepared extraction")
for item in prepared:
    label = f"Page {item['page']} · {item['kind']}" if item["page"] else item["kind"]
    st.success(f"{label}: {len(item['groups'])} group images, {item['rows']} financial rows")
    with st.expander(f"Preview {label} group images"):
        for group in item["groups"]:
            st.image(group["image"], caption=f"{group['id']} · {group['name']}", width="stretch")

if st.button("Run notebook model", type="primary"):
    st.session_state.pop("results", None)
    st.session_state.pop("results_signature", None)
    try:
        model, processor = load_model_cached()
        results = []
        st.session_state["results"] = results
        st.session_state["results_signature"] = signature
        st.session_state["results_complete"] = False
        total = sum(len(item["groups"]) for item in prepared)
        progress = st.progress(0, text="Starting extraction")
        done = 0
        for item in prepared:
            result = {"document": f"Bilan {item['kind'].title()}", "modele": MODEL_ID,
                      "prompt": settings[item["kind"]]["version"],
                      "page": item["page"], "groupes": []}
            results.append(result)
            for group in item["groups"]:
                progress.progress(done / total, text=f"{item['kind']} · {group['name']} ({done + 1}/{total})")
                data = infer_group(model, processor, group["image"], item["kind"],
                                   group["name"], group["expected_rows"])
                result["groupes"].append(data)
                done += 1
        progress.progress(1.0, text="Extraction complete")
        st.session_state["results_complete"] = True
    except Exception as exc:
        st.error(f"Extraction stopped: {exc}")

if st.session_state.get("results_signature") == signature:
    results = [item for item in st.session_state["results"] if item["groupes"]]
    if not results:
        st.stop()
    st.subheader("Results" if st.session_state.get("results_complete") else "Partial results")
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for i, result in enumerate(results, 1):
            filename = f"{result['document'].lower().replace(' ', '_')}_page_{result['page'] or i}.json"
            data = json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8")
            bundle.writestr(filename, data)
            with st.expander(filename):
                st.json(result)
                st.download_button("Download JSON", data, file_name=filename, mime="application/json", key=f"download_{i}")
    st.download_button("Download all results (.zip)", archive.getvalue(),
                       file_name="financial_extraction.zip", mime="application/zip")
