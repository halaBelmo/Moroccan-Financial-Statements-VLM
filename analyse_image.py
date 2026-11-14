#!/usr/bin/env python3
"""Bilan marocain (ACTIF/PASSIF) : segmentation OpenCV + extraction Groq.

Aucun numero de ligne, nombre de traits, rubrique ou position de page n'est fixe.
Fonctionne surtout avec les tableaux quadrilles. Le mode --prepare-only ne consomme
aucun token. Les pages non quadrillees sont signalees, pas extraites au hasard.

Usage:
    python analyse_bilan_dynamique.py --pdf data/bilan_group_2.pdf --prepare-only
    python analyse_bilan_dynamique.py --pdf data/bilan_group_2.pdf --types ACTIF,PASSIF
    python analyse_bilan_dynamique.py --pdf data/autre.pdf       # types auto par VLM
"""
from __future__ import annotations
import unicodedata
import argparse
import base64
import hashlib
import json
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
VERSION = "opencv_dynamic_v2"
CHAMPS = {
    "ACTIF": ["brut", "amortissements_provisions", "net_N", "net_N_1"],
    "PASSIF": ["net_N", "net_N_1"],
}

RUBRIQUES_ACTIF = {
    "IMMOBILISATIONS EN NON VALEUR": "A",
    "IMMOBILISATIONS INCORPORELLES": "B",
    "IMMOBILISATIONS CORPORELLES": "C",
    "IMMOBILISATIONS FINANCIERES": "D",
    "ECART DE CONVERSION ACTIF": "E",
    "STOCKS": "F",
    "CREANCES DE L ACTIF CIRCULANT": "G",
    "TITRES VAL DE PLACEMENT": "H",
}
RUBRIQUES_PASSIF = {
    "CAPITAUX PROPRES": "A",
    "CAPITAUX PROPRES ASSIMILES": "B",
    "DETTES DE FINANCEMENT": "C",
    "PROVIS DURABL": "D",
    "ECARTS DE CONVERSION PASSIF": "E",
    "DETTES DU PASSIF CIRCULANT": "F",
    "AUTRES PROVIS POUR RISQUES": "G",
    "ECART DE CONVERSION PASSIF E CIR": "H",
}
@dataclass
class Grille:
    limites: tuple[int, int, int, int]
    traits: list[int]
    verticales: list[int]
    debut_texte: int
    fin_libelle: int
    en_tete: int
    lignes: list[int]


def sauver(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def normaliser_texte(texte):
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(
        c for c in texte
        if not unicodedata.combining(c)
    )
    texte = texte.upper()
    texte = re.sub(r"[^A-Z0-9 ]", " ", texte)

    return " ".join(texte.split())

def lire(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def attribuer_rubriques(data):

    document = data["type_document"]

    referentiel = (
        RUBRIQUES_ACTIF
        if document == "ACTIF"
        else RUBRIQUES_PASSIF
    )

    rubrique_courante = None

    for ligne in data["lignes"]:

        poste = ligne["poste"]
        type_ligne = ligne["type_ligne"]
        texte = normaliser_texte(poste)

        if type_ligne == "total":
            ligne["rubrique"] = None
            rubrique_courante = None
            continue

        if type_ligne == "rubrique":

            # Chercher une lettre explicite (A) ... (I)
            codes = re.findall(r"\(([A-I])\)", poste.upper())
            code = codes[-1] if codes else None

            # La trésorerie ne possède pas de code A-I
            if "TRESORERIE" in texte:
                code = None

            elif code is None:
                # Rechercher dans le référentiel
                for nom, valeur in sorted(
                    referentiel.items(),
                    key=lambda item: len(item[0]),
                    reverse=True
                ):
                    if texte.startswith(nom):
                        code = valeur
                        break

            ligne["rubrique"] = code
            rubrique_courante = code

        elif type_ligne == "detail":
            ligne["rubrique"] = rubrique_courante

    return data

def empreinte(*values) -> str:
    h = hashlib.sha256()
    for v in values:
        h.update(v if isinstance(v, bytes) else str(v).encode("utf-8"))
    return h.hexdigest()


def groupes_proches(values: list[int], marge: int) -> list[int]:
    result = []
    for v in sorted(values):
        if not result or v - result[-1][-1] > marge:
            result.append([v])
        else:
            result[-1].append(v)
    return [round(float(np.median(g))) for g in result]


def detecter_grille(image: np.ndarray) -> tuple[Grille, np.ndarray]:
    """Lignes, colonnes et lignes vides detectees sans connaissance du formulaire."""
    gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gris.shape
    noir = cv2.adaptiveThreshold(
        gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 35, 13,
    )
    # Les noyaux suivent la taille de la page, non une grille TALIANI AUTO.
    hz = cv2.morphologyEx(noir, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(50, w // 7), 1)))
    vz = cv2.morphologyEx(noir, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(35, h // 8))))

    n, _, stats, _ = cv2.connectedComponentsWithStats(hz)
    segments = [(int(x), int(y), int(ww), int(hh))
                for x, y, ww, hh, _ in stats[1:]
                if ww >= w * .43 and hh <= max(15, h * .015)]
    if len(segments) < 7:
        raise ValueError("Tableau quadrille non detecte : format sans bordures ou scan de mauvaise qualite")

    # Les grands segments horizontaux delimiteurs du meme tableau.
    x0 = min(s[0] for s in segments)
    x1 = max(s[0] + s[2] for s in segments)
    ys = groupes_proches([y + hh // 2 for x, y, ww, hh in segments], max(5, h // 500))
    if len(ys) < 7 or (ys[-1] - ys[0]) < h * .20:
        raise ValueError("Grille trop petite ou nombre de lignes detectees insuffisant")

    n, _, vstats, _ = cv2.connectedComponentsWithStats(vz)
    vx = [int(x + ww // 2) for x, y, ww, hh, area in vstats[1:]
          if hh >= (ys[-1] - ys[0]) * .35 and x0 - 8 <= x <= x1 + 8]
    verticales = groupes_proches(vx, max(3, w // 600))
    if len(verticales) < 3:
        raise ValueError("Separations verticales insuffisantes pour identifier les colonnes")

    # Certains tableaux ont une petite colonne de lettres verticales a gauche.
    debut_texte = x0
    if len(verticales) >= 3 and verticales[1] - verticales[0] < (x1 - x0) * .16:
        debut_texte = verticales[1]
    fin_libelle = next((x for x in verticales if x > debut_texte + (x1 - x0) * .22), x1)

    # Heuristique d'en-tete : les colonnes Brut/Amort/Net commencent parfois
    # dans la deuxieme rangee (cas ACTIF). Sinon une rangee d'en-tete.
    def compte_verticales(yi):
        if yi + 1 >= len(ys):
            return 0
        mid = (ys[yi] + ys[yi + 1]) // 2
        return len(groupes_proches(np.flatnonzero(vz[mid, x0:x1]).tolist(), 4))

    en_tete = 2 if compte_verticales(1) >= compte_verticales(0) + 2 else 1

    # Retirer la grille pour distinguer les lignes de texte des lignes vides.
    texte = cv2.bitwise_and(noir, cv2.bitwise_not(cv2.bitwise_or(hz, vz)))
    lignes = []
    minimum = max(20, (fin_libelle - debut_texte) // 30)
    for i, (a, b) in enumerate(zip(ys, ys[1:])):
        if i < en_tete or b - a < 12:
            continue
        bande = texte[a + 5:b - 4, debut_texte + 6:fin_libelle - 6]
        if bande.size and cv2.countNonZero(bande) >= minimum:
            lignes.append(i)

    if not lignes:
        raise ValueError("Aucune ligne comptable detectee (le texte est peut-etre illisible)")

    return Grille((x0, ys[0], x1, ys[-1]), ys, verticales,
                  debut_texte, fin_libelle, en_tete, lignes), texte


def annoter(image: np.ndarray, g: Grille, chemin: Path):
    dessin = image.copy()
    x0, y0, x1, y1 = g.limites
    cv2.rectangle(dessin, (x0, y0), (x1, y1), (0, 120, 255), 4)
    for y in g.traits:
        cv2.line(dessin, (x0, y), (x1, y), (0, 160, 0), 2)
    for x in g.verticales:
        cv2.line(dessin, (x, y0), (x, y1), (255, 0, 0), 2)
    for i in g.lignes:
        cv2.putText(dessin, str(i), (max(0, x0 - 46), g.traits[i + 1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(chemin), dessin)


def creer_bloc(image: np.ndarray, g: Grille, indices: list[int], cible: Path):
    """Recompose 1 image de N lignes avec identifiants et entete reitere."""
    x0, _, x1, _ = g.limites
    x0 = max(0, g.debut_texte - 3)  # exclut les lettres verticales
    bande = image[g.traits[0]:g.traits[g.en_tete], x0:x1]
    # Repeter les noms de colonnes dans chaque image.
    hauteur = len(bande) + 14
    pieces = []
    for i in indices:
        part = image[g.traits[i]:g.traits[i + 1], x0:x1]
        pieces.append(part)
        hauteur += len(part) + 3
    canvas = np.full((hauteur + 6, x1 - x0 + 62, 3), 255, dtype=np.uint8)
    yy = 0
    canvas[yy:yy + len(bande), 62:62 + bande.shape[1]] = bande
    yy += len(bande) + 14
    for idx, piece in zip(indices, pieces):
        canvas[yy:yy + len(piece), 62:62 + piece.shape[1]] = piece
        cv2.putText(canvas, str(idx), (10, yy + max(19, len(piece) // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
        yy += len(piece) + 3
    cible.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cible), canvas)


def image_b64(image: np.ndarray):
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Encodage de l'image impossible")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def appel(client, chemin: Path, prompt: str, budget: int):
    from groq import RateLimitError
    b64 = base64.b64encode(chemin.read_bytes()).decode("ascii")
    try:
        rep = client.chat.completions.create(
            model=MODEL, max_completion_tokens=budget,
            response_format={"type": "json_object"}, reasoning_effort="none",
            temperature=0.1, messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            ]}],
        )
    except RateLimitError as exc:
        retry = getattr(getattr(exc, "response", None), "headers", {}).get("retry-after", "?")
        raise RuntimeError(f"Groq 429 : quota atteint. retry-after={retry}s. Le cache reste intact.") from exc
    texte = rep.choices[0].message.content or ""
    if rep.choices[0].finish_reason == "length":
        raise ValueError("Reponse tronquee par la limite de tokens")
    return json.loads(texte)


def analyse_entete(client, image: np.ndarray, cache: Path, rows_estimees: int):
    if cache.exists():
        return lire(cache)
    # Seul l'en-tete + les 3 premieres lignes : peu de texte a generer.
    h, w = image.shape[:2]
    miniature = image[:min(h, max(300, int(h * .23))), :]
    image_tmp = cache.with_suffix(".png")
    image_tmp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(image_tmp), miniature)
    prompt = ("Lis uniquement l'en-tete de cette page de bilan marocain. "
              "Retourne JSON: {\"type_document\":\"ACTIF ou PASSIF ou AUTRE\","
              "\"entreprise\":null,\"date_cloture\":null,\"header_rows\":1}. "
              "header_rows = nombre de lignes d'en-tete du TABLEAU (1, 2 ou 3), "
              "pas les postes. Utilise null si le nom ou la date est illisible. "
              "N'invente rien. JSON seul.")
    res = appel(client, image_tmp, prompt, 160)
    typ_brut = str(res.get("type_document", "AUTRE")).upper().strip()
    typ = next((val for val in ("PASSIF", "ACTIF") if val in typ_brut), "AUTRE")
    val = {"type_document": typ,
           "entreprise": res.get("entreprise") if isinstance(res.get("entreprise"), str) else None,
           "date_cloture": res.get("date_cloture") if isinstance(res.get("date_cloture"), str) else None,
           "header_rows": res.get("header_rows") if res.get("header_rows") in (1, 2, 3) else rows_estimees}
    sauver(cache, val)
    return val


def numerique(v):
    if v is None:
        return None
    if isinstance(v, bool):
        raise ValueError("Booleen invalide comme montant")
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().replace("\u00a0", " ").replace("\u202f", " ")
        if re.fullmatch(r"-?(?:\d+|\d{1,3}(?: \d{3})+)(?:[.,]\d+)?", s):
            return float(s.replace(" ", "").replace(",", "."))
    raise ValueError(f"Montant mal forme: {v!r}")


def prompt_bloc(typ: str, indices: list[int]):
    nums = ",".join(map(str, indices))
    colonnes = ("Brut, Amortissements, Net N, Net N-1" if typ == "ACTIF"
                else "Exercice N, Exercice N-1")
    fmt = ("[12,\"poste\",null,null,null,null]" if typ == "ACTIF"
           else "[12,\"poste\",null,null]")
    return (f"Transcris exactement {len(indices)} lignes visibles du tableau {typ}, "
            f"numerotees {nums}. Colonnes dans l'ordre: {colonnes}. "
            f'Retourne uniquement JSON compact de forme {{"l":[{fmt}]}}. '
            "Chaque ligne commence par son numero imprime dans la marge ajoutee, "
            "suivi du libelle et des montants sur la MEME rangee. "
            "N'ajoute aucune ligne. Respecte la casse des libelles (titres en MAJUSCULES). "
            "Cases vides/hachurees=null; jamais 0 a la place. "
            "Montants numeriques JSON (point decimal) et signe negatif preserve. "
            "Pas de calcul ni de commentaire. La bande superieure n'est PAS a extraire.")


def verifier_bloc(obj, typ, indices):
    l = obj.get("l") if isinstance(obj, dict) else None
    if not isinstance(l, list) or len(l) != len(indices):
        raise ValueError("Nombre de lignes VLM incorrect")
    ret = []
    for expected, raw in zip(indices, l):
        if not isinstance(raw, list) or len(raw) != len(CHAMPS[typ]) + 2:
            raise ValueError(f"Ligne {expected} : mauvais nombre de colonnes")
        if raw[0] != expected or not isinstance(raw[1], str) or not raw[1].strip():
            raise ValueError(f"Ligne {expected} : numero ou intitule incorrect")
        ret.append([raw[1].strip()] + [numerique(a) for a in raw[2:]])
    return ret


def extraire_bloc(client, typ, image, grille, indices, work, max_tokens):
    path = work / "blocs" / (f"{indices[0]:03d}_{indices[-1]:03d}.png")
    creer_bloc(image, grille, indices, path)
    p = prompt_bloc(typ, indices)
    key = empreinte(VERSION, MODEL, typ, max_tokens, path.read_bytes(), p)[:20]
    cache = work / "cache" / (key + ".json")
    if cache.exists():
        try:
            res = verifier_bloc(lire(cache), typ, indices)
            print(f"  Cache {indices[0]}-{indices[-1]}")
            return res
        except (ValueError, KeyError, TypeError):
            pass
    flag = work / "cache" / (key + ".split")
    if not flag.exists():
        print(f"  Groq {indices[0]}-{indices[-1]}")
        try:
            data = appel(client, path, p, max_tokens)
            res = verifier_bloc(data, typ, indices)
            sauver(cache, data)
            return res
        except (ValueError, json.JSONDecodeError) as exc:
            flag.parent.mkdir(parents=True, exist_ok=True)
            flag.write_text(str(exc), encoding="utf-8")
            print("  Reponse incomplete ou schema incorrect : sous-decoupage")
    if len(indices) <= 2:
        raise ValueError(f"Bloc {indices}: extraction VLM non valide, verifier image/source")
    cut = len(indices) // 2
    return (extraire_bloc(client, typ, image, grille, indices[:cut], work, max_tokens)
            + extraire_bloc(client, typ, image, grille, indices[cut:], work, max_tokens))


def sans_accents(s: str):
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def est_rubrique(nom: str):
    s = sans_accents(nom.strip())
    if re.search(r"\bTOTAL\b", s.upper()):
        return False
    if nom.lstrip().startswith(("*", "-", "•")):
        return False
    letters = [c for c in s if c.isalpha()]
    return bool(letters and sum(c.isupper() for c in letters) / len(letters) > .82)


def structurer(typ, extracted, meta, page):
    code = None
    resultat = []
    for idx, ligne in sorted(extracted.items()):
        poste, *nums = ligne
        normalise = sans_accents(poste).upper()
        total = bool(re.search(r"\bTOTAL\b", normalise))
        rubrique = est_rubrique(poste)
        fin = re.search(r"\(([A-I])\)\s*$", normalise)
        if rubrique:
            code = fin.group(1) if fin else None
        if "TRESORERIE" in normalise and rubrique:
            code = None
        kind = "total" if total else "rubrique" if rubrique else "detail"
        resultat.append({"source_page": page, "source_row": idx,
                         "poste": poste, "rubrique": None if total else code,
                         "type_ligne": kind, **dict(zip(CHAMPS[typ], nums))})
    # Les passifs CGNC peuvent donner le code (A) seulement dans TOTAL DES
    # CAPITAUX PROPRES (A) : le reporter sur les premieres lignes anterieures.
    if resultat:
        first_total = next((i for i, line in enumerate(resultat)
                            if line["type_ligne"] == "total" and
                            re.search(r"\(([A-I])\)\s*$", sans_accents(line["poste"]).upper())), None)
        if first_total is not None and all(item["rubrique"] is None for item in resultat[:first_total]):
            lettre = re.search(r"\(([A-I])\)\s*$", sans_accents(resultat[first_total]["poste"]).upper()).group(1)
            for item in resultat[:first_total]:
                item["rubrique"] = lettre
    return {"type_document": typ, "entreprise": meta.get("entreprise"),
            "date_cloture": meta.get("date_cloture"), "source_page": page,
            "lignes": resultat}


def controler(data):
    rapport = []
    totals = {}

    # 1. Identifier les totaux par leurs libellés
    for ligne in data["lignes"]:
        poste = sans_accents(ligne["poste"]).upper().strip()

        if re.search(r"\bTOTAL\s+GENERAL\b", poste):
            totals["GENERAL"] = ligne
        elif re.search(r"\bTOTAL\s+III\b", poste):
            totals["III"] = ligne
        elif re.search(r"\bTOTAL\s+II\b", poste):
            totals["II"] = ligne
        elif re.search(r"\bTOTAL\s+I\b", poste):
            totals["I"] = ligne

    document = data["type_document"]

    # 2. Vérifier chaque colonne comptable
    for field in CHAMPS[document]:

        manquants = [
            k for k in ("I", "II", "III", "GENERAL")
            if k not in totals
        ]

        if manquants:
            rapport.append(
                f"{field}: INCOMPLET | totaux absents: {manquants}"
            )
            continue

        # Dans l'ACTIF, TOTAL II et III peuvent avoir
        # une cellule d'amortissements réellement vide.
        if document == "ACTIF" and field == "amortissements_provisions":
            obligatoires = ("I", "GENERAL")
        else:
            obligatoires = ("I", "II", "GENERAL")

            if document == "ACTIF":
                obligatoires += ("III",)

        if any(totals[k].get(field) is None for k in obligatoires):
            rapport.append(
                f"{field}: INCOMPLET | montant obligatoire absent"
            )
            continue

        # 3. Additionner les sous-totaux
        somme = sum(
            (
                Decimal(str(totals[k][field]))
                for k in ("I", "II", "III")
                if totals[k].get(field) is not None
            ),
            Decimal("0")
        )

        total_general = Decimal(str(totals["GENERAL"][field]))
        ecart = total_general - somme

        statut = "OK" if abs(ecart) <= Decimal("0.01") else "ECART"

        cellules_vides = [
            k for k in ("I", "II", "III")
            if totals[k].get(field) is None
        ]

        message = (
            f"{field}: {statut} "
            f"| somme={somme:.2f} "
            f"| total={total_general:.2f} "
            f"| ecart={ecart:.2f}"
        )

        if cellules_vides:
            message += (
                f" | cellules vides considérées comme 0 "
                f"pour ce contrôle: {cellules_vides}"
            )

        rapport.append(message)

    return rapport


def comparer(actif, passif):
    def general(obj):
        return next((r for r in obj["lignes"] if "TOTAL GENERAL" in sans_accents(r["poste"]).upper()), None)
    a, p = general(actif), general(passif)
    if not (a and p):
        return ["ACTIF/PASSIF: total general introuvable"]
    result = []
    for k in ("net_N", "net_N_1"):
        if a.get(k) is None or p.get(k) is None:
            result.append(f"{k}: INCOMPLET")
        else:
            diff = Decimal(str(a[k])) - Decimal(str(p[k]))
            result.append(f"{k}: {'OK' if abs(diff) <= Decimal('0.01') else 'ECART'} | ecart={diff}")
    return result


def main():
    ap = argparse.ArgumentParser(description="Extraction dynamique OpenCV + Groq de bilans quadrilles")
    ap.add_argument("--pdf", type=Path, default=ROOT / "data" / "bilan_group_2.pdf")
    ap.add_argument("--document", choices=["ALL", "ACTIF", "PASSIF"], default="ALL")
    ap.add_argument("--types", help="Types par page, ex. ACTIF,PASSIF (economise la detection VLM)")
    ap.add_argument("--prepare-only", action="store_true", help="Segmentation OpenCV uniquement, 0 appel API")
    ap.add_argument("--rows", type=int, default=9, help="Lignes par image Groq (defaut 9)")
    ap.add_argument("--header-rows", type=int, choices=[1, 2, 3],
                    help="Correction manuelle si OpenCV se trompe sur le nombre de lignes d'entete")
    ap.add_argument("--max-tokens", type=int, default=800)
    args = ap.parse_args()
    if not 2 <= args.rows <= 12 or not 200 <= args.max_tokens <= 950:
        ap.error("--rows 2..12 et --max-tokens 200..950 requis")
    pdf = args.pdf.expanduser().resolve()
    if not pdf.exists():
        ap.error(f"PDF introuvable : {pdf}")
    keypdf = empreinte(pdf.read_bytes())[:12]
    output = ROOT / "outputs" / f"{pdf.stem}_{keypdf}"
    images = ROOT / "images" / f"{pdf.stem}_{keypdf}"
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    load_dotenv(ROOT / ".env")
    types_override = [x.strip().upper() for x in args.types.split(",")] if args.types else []
    client = None
    documents = {}

    with pymupdf.open(pdf) as doc:
        if types_override and len(types_override) != len(doc):
            ap.error("--types doit donner un type par page, ex. ACTIF,PASSIF")
        for page_number in range(len(doc)):
            pix = doc[page_number].get_pixmap(matrix=pymupdf.Matrix(200 / 72, 200 / 72), alpha=False)
            img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).copy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            page_path = images / f"page_{page_number+1:02d}.png"
            cv2.imwrite(str(page_path), img)
            try:
                grille, masque = detecter_grille(img)
            except ValueError as exc:
                print(f"Page {page_number+1}: ignoree ({exc})")
                continue
            if args.header_rows is not None:
                grille.en_tete = args.header_rows
                grille.lignes = [i for i in grille.lignes if i >= grille.en_tete]
            annoter(img, grille, images / f"page_{page_number+1:02d}_debug.png")
            print(f"Page {page_number+1}: {len(grille.traits)} traits, {len(grille.lignes)} lignes non vides, "
                  f"{len(grille.verticales)} separations verticales")
            if args.prepare_only:
                typ = types_override[page_number] if types_override else "INCONNU"
                meta = {"type_document": typ, "entreprise": None, "date_cloture": None,
                        "header_rows": grille.en_tete}
            elif types_override:
                typ = types_override[page_number]
                meta = {"type_document": typ, "entreprise": None,
                        "date_cloture": None, "header_rows": grille.en_tete}
            else:
                if client is None:
                    from groq import Groq
                    cle = os.getenv("GROQ_API_KEY")
                    if not cle:
                        ap.error("GROQ_API_KEY absent du fichier .env")
                    client = Groq(api_key=cle, max_retries=0)
                meta = analyse_entete(client, img, output / f"page_{page_number+1:02d}_meta.json", grille.en_tete)
                typ = meta["type_document"]
                if meta["header_rows"] != grille.en_tete:
                    print(f"  Note : le VLM suggere {meta['header_rows']} ligne(s) d'entete, "
                          f"OpenCV en detecte {grille.en_tete}; on conserve OpenCV. "
                          "Inspecte l'image --prepare-only si besoin.")
            if typ == "AUTRE":
                print(f"Page {page_number+1}: ni ACTIF ni PASSIF, ignoree")
                continue
            if not args.prepare_only and typ not in CHAMPS:
                raise ValueError(f"Page {page_number+1}: classification inconnue: {typ}")
            if args.document != "ALL" and typ != args.document and not args.prepare_only:
                continue
            work = output / f"page_{page_number+1:02d}"
            blocs = []
            for start in range(0, len(grille.lignes), args.rows):
                indices = grille.lignes[start:start+args.rows]
                path = work / "blocs" / f"{indices[0]:03d}_{indices[-1]:03d}.png"
                creer_bloc(img, grille, indices, path)
                blocs.append(indices)
            sauver(work / "segmentation.json", {
                "page": page_number+1, "table_box": grille.limites,
                "horizontal_lines": grille.traits, "vertical_lines": grille.verticales,
                "header_rows": grille.en_tete, "data_rows": grille.lignes,
                "blocs": blocs,
            })
            print(f"  {len(blocs)} blocs produits : {work / 'blocs'}")
            if args.prepare_only:
                continue
            if client is None:
                from groq import Groq
                cle = os.getenv("GROQ_API_KEY")
                if not cle:
                    ap.error("GROQ_API_KEY absent du fichier .env")
                client = Groq(api_key=cle, max_retries=0)
            extracted = {}
            try:
                for i, indices in enumerate(blocs, 1):
                    found = extraire_bloc(client, typ, img, grille, indices, work, args.max_tokens)
                    extracted.update(zip(indices, found))
                    # Sauvegarder aussi les VALEURS deja extraites, pas seulement leurs indices.
                    sauver(work / "partiel.json", {"type_document": typ,
                           "lignes": {str(k): v for k, v in extracted.items()}})
                    print(f"  [{i}/{len(blocs)}] {len(extracted)}/{len(grille.lignes)} lignes")
            except (RuntimeError, ValueError) as exc:
                print(f"Page {page_number+1}: extraction interrompue: {exc}")
                print("Les caches et le fichier partiel ont ete conserves; relance la meme commande.")
                return 2
            resultat = structurer(typ, extracted, meta, page_number+1)
            resultat = attribuer_rubriques(resultat)

            sauver(output / f"{typ.lower()}_page_{page_number+1}.json", resultat)
            report = controler(resultat)
            (output / f"controle_{typ.lower()}_page_{page_number+1}.txt").write_text(
                "\n".join(report) + "\n", encoding="utf-8")
            print(f"{typ} : {len(resultat['lignes'])} lignes JSON; controles: {report}")
            documents.setdefault(typ, []).append(resultat)

    if args.prepare_only:
        print("Decoupage pret : examine les fichiers *_debug.png, segmentation.json et blocs/*.png")
        return 0
    if len(documents.get("ACTIF", [])) == 1 and len(documents.get("PASSIF", [])) == 1:
        checks = comparer(documents["ACTIF"][0], documents["PASSIF"][0])
        (output / "controle_actif_passif.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")
        print("Controle ACTIF = PASSIF:", *checks, sep="\n  ")
    print(f"Resultats disponibles dans : {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
