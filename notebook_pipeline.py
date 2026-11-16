"""Prepare notebook-style group images and run the notebook's local Qwen model."""

from __future__ import annotations

import json
import re

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageFont

from analyse_image import detecter_grille
from notebook_config import MODEL_ID, notebook_settings

HEADERS = {
    "ACTIF": ["POSTE", "BRUT", "AMORTISSEMENTS", "NET", "NET N-1"],
    "PASSIF": ["POSTE", "EXERCICE", "EXERCICE PRECEDENT"],
}
PALETTE = [(135, 206, 250), (255, 160, 190), (130, 220, 150),
           (255, 190, 100), (190, 160, 240), (110, 220, 220)]
REQUIRED = {"id_ligne", "poste", "brut", "amortissements", "net", "net_precedent"}


def render_pdf(data: bytes) -> list[Image.Image]:
    with pymupdf.open(stream=data, filetype="pdf") as document:
        if document.is_encrypted:
            raise ValueError("Password-protected PDFs are not supported.")
        if len(document) > 20:
            raise ValueError("This interface accepts PDFs of up to 20 pages.")
        return [Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                for page in document
                for pix in [page.get_pixmap(matrix=pymupdf.Matrix(300 / 72, 300 / 72), alpha=False)]]


def _columns(grille, kind: str) -> list[int]:
    left = grille.debut_texte
    right = grille.limites[2]
    boundaries = sorted(x for x in grille.verticales if left - 8 <= x <= right + 8)
    boundaries = [left] + [x for x in boundaries if x > left + 8 and x < right - 8] + [right]
    wanted = len(HEADERS[kind]) + 1
    if len(boundaries) != wanted:
        raise ValueError(
            f"Expected {wanted} column borders for {kind}, found {len(boundaries)}. "
            "Check that this is a gridded Moroccan balance sheet."
        )
    return boundaries


def _make_group(page: np.ndarray, rows: list[tuple[int, int]], borders: list[int],
                kind: str) -> Image.Image:
    # The notebook colors each row, then adds column labels. PASSIF V5
    # additionally separates the three columns with 110-pixel white gaps.
    top, bottom = rows[0][0], rows[-1][1]
    crop = page[top:bottom, borders[0]:borders[-1]].copy()
    for i, (a, b) in enumerate(rows):
        y0, y1 = a - top, b - top
        if y1 <= y0:
            continue
        color = np.array(PALETTE[i % len(PALETTE)], dtype=np.float32)
        crop[y0:y1] = (crop[y0:y1].astype(np.float32) * .68 + color * .32).astype(np.uint8)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    columns = [rgb[:, borders[i] - borders[0] + 2:borders[i + 1] - borders[0] - 2]
               for i in range(len(borders) - 1)]
    if any(column.shape[1] < 20 for column in columns):
        raise ValueError("A detected column is too narrow to read.")
    gap = 110 if kind == "PASSIF" else 0
    margin, header_height = 20, 80
    width = sum(c.shape[1] for c in columns) + gap * (len(columns) - 1) + margin * 2
    canvas = Image.new("RGB", (width, rgb.shape[0] + header_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    x = margin
    for title, column in zip(HEADERS[kind], columns):
        canvas.paste(Image.fromarray(column), (x, header_height))
        center = x + column.shape[1] // 2
        box = draw.textbbox((0, 0), title, font=font)
        draw.text((center - (box[2] - box[0]) // 2, 25), title, fill="black", font=font)
        x += column.shape[1] + gap
    return canvas


def prepare_page(image: Image.Image, kind: str) -> dict:
    settings = notebook_settings()[kind]
    page = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    grille, _ = detecter_grille(page)
    grouped_rows = sum(count for _, count in settings["groups"].values())
    expected = grouped_rows + (2 if kind == "ACTIF" else 0)
    actual = len(grille.lignes)
    if actual != expected:
        raise ValueError(
            f"{kind}: detected {actual} financial rows; the notebook expects {expected}. "
            "Check the page and table segmentation before running the model."
        )
    borders = _columns(grille, kind)
    rows = [(grille.traits[i], grille.traits[i + 1]) for i in grille.lignes]
    groups = []
    offset = 2 if kind == "ACTIF" else 0
    for group_id, (name, count) in settings["groups"].items():
        groups.append({
            "id": group_id, "name": name, "expected_rows": count,
            "image": _make_group(page, rows[offset:offset + count], borders, kind),
        })
        offset += count
    return {"kind": kind, "rows": actual, "groups": groups}


def load_model():
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for the notebook's 4-bit Qwen model.")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=quant, device_map="auto", dtype=dtype,
        attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)
    model.eval()
    return model, processor


def infer_group(model, processor, image: Image.Image, kind: str,
                group_name: str, expected_rows: int) -> dict:
    import torch
    from qwen_vl_utils import process_vision_info

    prompt = notebook_settings()[kind]["prompt"](group_name, expected_rows)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image.convert("RGB")},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2048, do_sample=False)
    trimmed = [ids[len(input_ids):] for input_ids, ids in zip(inputs.input_ids, output)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip(), flags=re.IGNORECASE).strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model response is not valid JSON: {clean[:300]}") from exc
    validate_group(data, kind, group_name, expected_rows)
    return data


def validate_group(data: dict, kind: str, name: str, expected: int) -> None:
    if not isinstance(data, dict) or data.get("groupe") != name:
        raise ValueError(f"{kind} / {name}: incorrect group name in model output.")
    lines = data.get("lignes")
    if not isinstance(lines, list) or len(lines) != expected:
        raise ValueError(f"{kind} / {name}: expected {expected} output rows, got {len(lines) if isinstance(lines, list) else 'no list'}.")
    required = REQUIRED if kind == "ACTIF" else {"id_ligne", "poste", "exercice", "exercice_precedent"}
    for number, line in enumerate(lines, 1):
        if not isinstance(line, dict) or set(line) != required or line.get("id_ligne") != number:
            raise ValueError(f"{kind} / {name}: invalid row {number} in model output.")
        if line["poste"] is not None and (not isinstance(line["poste"], str) or not line["poste"].strip()):
            raise ValueError(f"{kind} / {name}: invalid label in row {number}.")
        for field in required - {"id_ligne", "poste"}:
            value = line[field]
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{kind} / {name}: {field} in row {number} must be text or null.")
    if lines[0]["poste"] != name:
        raise ValueError(f"{kind} / {name}: group title missing from first row.")
