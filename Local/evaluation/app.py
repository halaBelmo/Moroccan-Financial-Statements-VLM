
import json
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

from jsonschema import Draft7Validator
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent
LOCAL = BASE.parent

RESULTS = BASE / "results"
FIGURES = BASE / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

GT = (
    LOCAL.parent
    / "evaluation"
    / "ground_truth"
    / "amounts.json"
)

FILES = {
    "ACTIF": "bilan_actif_7b_v7.json",
    "PASSIF": "bilan_passif_7b_v5_v13.json"
}

COUNTS = {
    "ACTIF": [4, 5, 8, 5, 3, 1, 6, 8, 2, 2, 1, 4, 1, 1],
    "PASSIF": [10, 1, 3, 3, 3, 3, 1, 9, 2, 2, 1, 4, 1, 1]
}

ROWS = {
    "ACTIF": list(range(2, 53)),
    "PASSIF": (
        list(range(1, 15))
        + list(range(19, 22))
        + list(range(26, 53))
    )
}

COLS = {
    "ACTIF": [
        "brut",
        "amortissements",
        "net",
        "net_precedent"
    ],
    "PASSIF": [
        "exercice",
        "exercice_precedent"
    ]
}

REFCOLS = {
    "ACTIF": [
        "brut",
        "amortissements_provisions",
        "net_N",
        "net_N_1"
    ],
    "PASSIF": [
        "net_N",
        "net_N_1"
    ]
}

HEADS = {
    "ACTIF": [
        "IMMOBILISATIONS EN NON VALEUR (A)",
        "IMMOBILISATIONS INCORPORELLES (B)",
        "IMMOBILISATIONS CORPORELLES (C)",
        "IMMOBILISATIONS FINANCIERES (D)",
        "ECART DE CONVERSION-ACTIF (E)",
        "TOTAL I (A+B+C+D+E)",
        "STOCKS (F)",
        "CREANCES DE L'ACTIF CIRCULANT (G)",
        "TITRES & VAL. DE PLACEMENT (H)",
        "ECART DE CONVERSION-ACTIF (E.C) (I)",
        "TOTAL II (F+G+H+I)",
        "TRESORERIE - ACTIF",
        "TOTAL III",
        "TOTAL GENERAL (I+II+III)"
    ],
    "PASSIF": [
        "CAPITAUX PROPRES",
        "TOTAL DES CAPITAUX PROPRES (A)",
        "CAPITAUX PROPRES ASSIMILES (B)",
        "DETTES DE FINANCEMENT (C)",
        "PROVIS. DURABL. PR RISQUES & CHARGES (D)",
        "ECARTS DE CONVERSION - PASSIF (E)",
        "TOTAL I (A+B+C+D+E)",
        "DETTES DU PASSIF CIRCULANT (F)",
        "AUTRES PROVIS POUR RISQUES & CHARGES (G)",
        "ECART DE CONVERSION - PASSIF (E.CIR (H)",
        "TOTAL II (F+G+H)",
        "TRESORERIE - PASSIF",
        "TOTAL III",
        "TOTAL GENERAL (I+II+III)"
    ]
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def read(path):
    with path.open(encoding="utf-8-sig") as f:
        return json.load(f)


def save(name, data):
    with open(RESULTS / name, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
            default=str
        )


def number(value):

    if value is None:
        return None

    result = Decimal(
        re.sub(r"\s+", "", str(value))
        .replace(",", ".")
    )

    if not result.is_finite():
        raise ValueError("Nonfinite amount")

    return result


def pct(a, b):
    return 100 * a / b if b else 0


# ==========================================
# 3. TEST 01 - JSON VALIDITY
# ==========================================

def schema(document):

    properties = {
        "id_ligne": {
            "type": "integer",
            "minimum": 1
        },
        "poste": {
            "type": "string",
            "minLength": 1
        }
    }

    for col in COLS[document]:

        properties[col] = {
            "type": ["number", "string", "null"]
        }

    line = {
        "type": "object",
        "required": list(properties),
        "properties": properties
    }

    group = {
        "type": "object",
        "required": ["groupe", "lignes"],
        "properties": {
            "groupe": {"type": "string"},
            "lignes": {
                "type": "array",
                "minItems": 1,
                "items": line
            }
        }
    }

    return {
        "type": "object",
        "required": ["document", "modele", "groupes"],
        "properties": {
            "document": {"type": "string"},
            "modele": {"type": "string"},
            "groupes": {
                "type": "array",
                "minItems": 1,
                "items": group
            }
        }
    }


def main():

    validity = []
    completeness = []
    accuracy = []
    controls = []
    balances = []
    docs = {}

    print("\n===== TEST 01: JSON VALIDITY =====")

    for doc, filename in FILES.items():

        result = {
            "document": doc,
            "json_valide": False,
            "schema_valide": False,
            "nombre_erreurs": 0,
            "erreurs": []
        }

        try:

            data = read(LOCAL / filename)

            result["json_valide"] = True

            validator = Draft7Validator(schema(doc))

            result["erreurs"] = [
                e.message
                for e in validator.iter_errors(data)
            ]

            result["schema_valide"] = (
                not result["erreurs"]
            )

            docs[doc] = data

        except (OSError, ValueError) as e:

            result["erreurs"] = [str(e)]

        result["nombre_erreurs"] = len(
            result["erreurs"]
        )

        validity.append(result)

        print(doc, result["schema_valide"])

    save("test_01_json.json", validity)

    if not all(r["schema_valide"] for r in validity):

        raise SystemExit(
            "Invalid JSON: check test_01_json.json"
        )

    ground_truth = read(GT)

    # ======================================
    # 4. TEST 02 - COMPLETENESS
    # ======================================

    for doc, data in docs.items():

        groups = data["groupes"]
        counts = COUNTS[doc]
        columns = COLS[doc]

        if [g["groupe"] for g in groups] != HEADS[doc]:

            raise SystemExit(
                f"{doc}: review group alignment."
            )

        tp = fp = fn = 0

        duplicates = []
        aligned = {}

        offset = 0

        for group, count in zip(groups, counts):

            ids = Counter(
                line["id_ligne"]
                for line in group["lignes"]
            )

            expected = set(range(1, count + 1))
            actual = set(ids)

            tp += len(expected & actual)
            fn += len(expected - actual)

            fp += sum(
                v for k, v in ids.items()
                if k not in expected
            )

            fp += sum(
                ids[k] - 1
                for k in expected & actual
            )

            duplicates += [
                {
                    "groupe": group["groupe"],
                    "id_ligne": k
                }
                for k, v in ids.items()
                if v > 1
            ]

            for line in group["lignes"]:

                k = line["id_ligne"]

                if k in expected and ids[k] == 1:

                    source_row = ROWS[doc][offset + k - 1]

                    aligned[source_row] = line

            offset += count

        precision = pct(tp, tp + fp)
        recall = pct(tp, tp + fn)

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall else 0
        )

        completeness.append({
            "document": doc,
            "lignes_attendues": sum(counts),
            "lignes_extraites": sum(
                len(g["lignes"]) for g in groups
            ),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "doublons": duplicates,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })

        # ==================================
        # 5. TEST 03 - AMOUNT ACCURACY
        # ==================================

        reference = ground_truth[doc]

        correct = 0
        total = 0
        amounts = 0
        good = 0

        errors = []

        if not isinstance(reference, dict):

            raise SystemExit(
                f"{doc}: unexpected ground truth format"
            )

        for row in ROWS[doc]:

            ref = reference.get(str(row), {})
            line = aligned.get(row)

            for i, col in enumerate(columns):

                if isinstance(ref, list):
                    value = ref[i]
                else:
                    value = ref.get(REFCOLS[doc][i])

                expected = number(value)

                total += 1

                if expected is not None:
                    amounts += 1

                try:

                    actual = (
                        number(line[col])
                        if line is not None else None
                    )

                    valid = line is not None

                except (ValueError, InvalidOperation):

                    actual = line[col]
                    valid = False

                if valid and actual == expected:

                    correct += 1

                    if expected is not None:
                        good += 1

                else:

                    errors.append({
                        "source_row": row,
                        "colonne": col,
                        "attendu": expected,
                        "extrait": actual,
                        "ligne_absente": line is None
                    })

        if amounts == 0:

            raise SystemExit(
                f"{doc}: no annotated amounts found"
            )

        accuracy.append({
            "document": doc,
            "cellules_testees": total,
            "cellules_correctes": correct,
            "cellules_incorrectes": total - correct,
            "accuracy_globale": pct(correct, total),
            "montants_testes": amounts,
            "montants_corrects": good,
            "accuracy_montants": pct(good, amounts),
            "erreurs": errors
        })

        # ==================================
        # 6. TEST 05 - FINANCIAL CONSISTENCY
        # ==================================

        if fp or fn:

            raise SystemExit(
                f"{doc}: review missing or extra rows"
            )

        def val(line, col):

            return number(line[col]) or Decimal(0)

        def check(name, col, parts, total_line):

            calculated = sum(
                (val(line, col) for line in parts),
                Decimal(0)
            )

            reported = val(total_line, col)

            controls.append({
                "document": doc,
                "controle": name,
                "colonne": col,
                "somme_calculee": calculated,
                "total_extrait": reported,
                "ecart": calculated - reported,
                "coherent": calculated == reported
            })

        # Rubric totals vs details

        for group in groups:

            lines = group["lignes"]

            if len(lines) > 1:

                for col in columns:

                    check(
                        group["groupe"],
                        col,
                        lines[1:],
                        lines[0]
                    )

        # Total I + II + III

        sections = [
            g["lignes"][0]
            for g in groups
            if re.match(
                r"^TOTAL (I|II|III)(?:\s|$)",
                g["groupe"]
            )
        ]

        for col in columns:

            check(
                "I + II + III",
                col,
                sections,
                groups[-1]["lignes"][0]
            )

    save("test_02_completeness.json", completeness)

    save("test_03_accuracy.json", accuracy)

    # ======================================
    # 7. TEST 04 - ACCOUNTING BALANCE
    # ======================================

    periods = [
        ("N", "net", "exercice"),
        ("N-1", "net_precedent", "exercice_precedent")
    ]

    for period, a, p in periods:

        actif = number(
            docs["ACTIF"]["groupes"][-1]["lignes"][0][a]
        )

        passif = number(
            docs["PASSIF"]["groupes"][-1]["lignes"][0][p]
        )

        ok = (
            actif is not None
            and passif is not None
            and actif == passif
        )

        balances.append({
            "exercice": period,
            "total_actif": actif,
            "total_passif": passif,
            "ecart": (
                actif - passif
                if actif is not None and passif is not None
                else None
            ),
            "equilibre": ok
        })

    balance_rate = pct(
        sum(x["equilibre"] for x in balances),
        len(balances)
    )

    successful_controls = sum(
        x["coherent"] for x in controls
    )

    consistency_rate = pct(
        successful_controls,
        len(controls)
    )

    save("test_04_balance.json", {
        "controles": balances,
        "taux_equilibre": balance_rate
    })

    save("test_05_consistency.json", {
        "controles": controls,
        "total": len(controls),
        "reussis": successful_controls,
        "taux_coherence": consistency_rate
    })

    # ======================================
    # 8. TEST 06 - ERROR ANALYSIS
    # ======================================

    tp = sum(x["TP"] for x in completeness)
    fp = sum(x["FP"] for x in completeness)
    fn = sum(x["FN"] for x in completeness)

    correct = sum(
        x["montants_corrects"] for x in accuracy
    )

    total = sum(
        x["montants_testes"] for x in accuracy
    )

    cells = sum(
        x["cellules_correctes"] for x in accuracy
    )

    allcells = sum(
        x["cellules_testees"] for x in accuracy
    )

    errors = {
        "Structure": sum(
            x["nombre_erreurs"] for x in validity
        ),
        "Missing rows": fn,
        "Extra rows": fp,
        "Incorrect cells": allcells - cells,
        "Balance failures": sum(
            not x["equilibre"] for x in balances
        ),
        "Consistency failures": (
            len(controls) - successful_controls
        )
    }

    metadata = [
        {
            "document": doc,
            "champ": field
        }
        for doc, data in docs.items()
        for field in ("entreprise", "date_cloture")
        if not data.get(field)
    ]

    save("test_06_errors.json", {
        "erreurs": errors,
        "total_anomalies": sum(errors.values()),
        "metadonnees_manquantes": metadata,
        "exactitude_textuelle": "Not fully evaluated"
    })

    # ======================================
    # 9. TEST 07 - INFERENCE LATENCY
    # ======================================

    notebook = read(
        LOCAL / "Local_extraction.ipynb"
    )

    times = {}

    for doc, indices in [
        ("ACTIF", [16, 17]),
        ("PASSIF", [25])
    ]:

        values = []

        for i in indices:

            cell = notebook["cells"][i]

            output = "\n".join(
                "".join(o.get("text", []))
                for o in cell.get("outputs", [])
            )

            matches = re.findall(
                r"Temps total\s*:\s*([0-9.]+)\s*secondes",
                output
            )

            if not matches:

                raise SystemExit(
                    "Recorded latency not found in notebook"
                )

            values.append(
                Decimal(matches[-1])
            )

        times[doc] = float(sum(values))

    latency = sum(times.values())

    save("test_07_latency.json", {
        "secondes": times,
        "total_secondes": round(latency, 2),
        "scope": "Recorded inference only"
    })

    # ======================================
    # 10. GLOBAL PERFORMANCE METRICS
    # ======================================

    precision = pct(tp, tp + fp)

    recall = pct(tp, tp + fn)

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall else 0
    )

    metrics = {
        "JSON validity": pct(
            sum(x["schema_valide"] for x in validity),
            len(validity)
        ),
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Amount accuracy": pct(correct, total),
        "Cell accuracy": pct(cells, allcells),
        "Accounting balance": balance_rate,
        "Financial consistency": consistency_rate
    }

    save("summary.json", {
        "metrics": metrics,
        "latency_seconds": round(latency, 2),
        "metadata_missing": len(metadata),
        "scope": (
            "One financial statement. "
            "Label accuracy not fully tested."
        )
    })

    print("\n===== LOCAL VLM EVALUATION =====")

    for name, score in metrics.items():

        print(f"{name}: {score:.2f}%")

    print(f"\nInference latency: {latency:.2f} seconds")

    print("\nAll results saved in:", RESULTS)

    dashboard(
        metrics,
        completeness,
        accuracy,
        balances,
        successful_controls,
        len(controls),
        errors,
        metadata,
        times,
        latency
    )


# ==========================================
# 11. MODERN DASHBOARD
# ==========================================

def dashboard(
    metrics,
    completeness,
    accuracy,
    balances,
    cs,
    ncontrols,
    errors,
    metadata,
    times,
    latency
):

    BG = "#0B1220"
    CARD = "#142035"
    BORDER = "#283952"

    TEXT = "#F1F5F9"
    MUTED = "#9BABBF"

    BLUE = "#38BDF8"
    GREEN = "#34D399"
    PURPLE = "#A78BFA"
    ORANGE = "#FBBF24"
    RED = "#FB7185"

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "text.color": TEXT,
        "toolbar": "None"
    })

    fig = plt.figure(
        figsize=(16, 9),
        facecolor=BG
    )

    # ======================================
    # PANEL COMPONENT
    # ======================================

    def panel(x, y, w, h, title=None):

        ax = fig.add_axes([x, y, w, h])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_navigate(False)

        ax.add_patch(
            FancyBboxPatch(
                (0.002, 0.002),
                0.996,
                0.996,
                boxstyle=(
                    "round,pad=0,rounding_size=0.025"
                ),
                facecolor=CARD,
                edgecolor=BORDER,
                linewidth=1
            )
        )

        if title:

            ax.text(
                0.055, 0.89,
                title,
                fontsize=13,
                weight="bold",
                va="center"
            )

        return ax

    # ======================================
    # PROGRESS BAR
    # ======================================

    def bar(ax, y, name, value, color):

        ax.text(
            0.06, y,
            name,
            fontsize=10,
            va="center"
        )

        ax.text(
            0.94, y,
            f"{value:.1f}%",
            ha="right",
            va="center",
            fontsize=10,
            color=color,
            weight="bold"
        )

        ax.plot(
            [0.06, 0.94],
            [y - 0.075] * 2,
            color=BORDER,
            lw=5,
            solid_capstyle="round"
        )

        ax.plot(
            [0.06, 0.06 + 0.88 * min(value / 100, 1)],
            [y - 0.075] * 2,
            color=color,
            lw=5,
            solid_capstyle="round"
        )

    # ======================================
    # HEADER
    # ======================================

    fig.text(
        0.035, 0.945,
        "LOCAL VLM / PERFORMANCE ANALYTICS",
        fontsize=23,
        weight="bold"
    )

    fig.text(
        0.035, 0.910,
        "Qwen2.5-VL-7B-Instruct  |  Kaggle  |  TALIANI AUTO",
        fontsize=11,
        color=MUTED
    )

    fig.text(
        0.965, 0.945,
        "LOCAL INFERENCE",
        ha="right",
        fontsize=11,
        color=BLUE,
        weight="bold"
    )

    # ======================================
    # KPI CARDS
    # ======================================

    tp = sum(x["TP"] for x in completeness)

    expected = sum(
        x["lignes_attendues"]
        for x in completeness
    )

    good = sum(
        x["montants_corrects"]
        for x in accuracy
    )

    total = sum(
        x["montants_testes"]
        for x in accuracy
    )

    kpis = [
        (
            "COMPLETENESS",
            f"{metrics['Recall']:.0f}%",
            f"{tp}/{expected} accounting rows",
            BLUE
        ),
        (
            "AMOUNT ACCURACY",
            f"{metrics['Amount accuracy']:.0f}%",
            f"{good}/{total} correct amounts",
            GREEN
        ),
        (
            "CONSISTENCY",
            f"{cs}/{ncontrols}",
            "Accounting controls",
            PURPLE
        ),
        (
            "INFERENCE TIME",
            f"{int(latency // 60)}m {int(latency % 60)}s",
            "Recorded inference",
            ORANGE
        )
    ]

    for i, (title, value, sub, color) in enumerate(kpis):

        ax = panel(
            0.035 + i * 0.237,
            0.75,
            0.22,
            0.13
        )

        ax.text(
            0.07, 0.76,
            title,
            fontsize=10,
            color=MUTED
        )

        ax.text(
            0.07, 0.37,
            value,
            fontsize=26,
            color=color,
            weight="bold"
        )

        ax.text(
            0.07, 0.13,
            sub,
            fontsize=9,
            color=MUTED
        )

    # ======================================
    # MODEL EVALUATION
    # ======================================

    ax = panel(
        0.035, 0.39,
        0.455, 0.33,
        "Model evaluation"
    )

    names = [
        "JSON validity",
        "Precision",
        "Recall",
        "F1-score"
    ]

    for i, name in enumerate(names):

        bar(
            ax,
            0.70 - i * 0.17,
            name,
            metrics[name],
            GREEN if metrics[name] == 100 else ORANGE
        )

    # ======================================
    # INFERENCE LATENCY
    # ======================================

    ax = panel(
        0.51, 0.39,
        0.455, 0.33,
        "Inference latency"
    )

    for i, (doc, value) in enumerate(times.items()):

        y = 0.67 - i * 0.27

        ax.text(
            0.06, y,
            doc,
            fontsize=11
        )

        ax.text(
            0.94, y,
            f"{value:.2f} s",
            ha="right",
            fontsize=12,
            color=BLUE,
            weight="bold"
        )

        ax.plot(
            [0.06, 0.94],
            [y - 0.10] * 2,
            color=BORDER,
            lw=9,
            solid_capstyle="round"
        )

        ratio = (
            value / max(times.values())
            if max(times.values()) else 0
        )

        ax.plot(
            [0.06, 0.06 + 0.88 * ratio],
            [y - 0.10] * 2,
            color=[BLUE, PURPLE][i],
            lw=9,
            solid_capstyle="round"
        )

    ax.text(
        0.06, 0.10,
        "Excludes model loading and preprocessing",
        fontsize=9,
        color=MUTED
    )

    # ======================================
    # FINANCIAL AMOUNTS
    # ======================================

    ax = panel(
        0.035, 0.075,
        0.295, 0.285,
        "Financial amount accuracy"
    )

    for i, row in enumerate(accuracy):

        y = 0.65 - i * 0.27

        ax.text(
            0.07, y,
            row["document"],
            fontsize=11
        )

        ax.text(
            0.93, y,
            f"{row['montants_corrects']}/"
            f"{row['montants_testes']}",
            ha="right",
            fontsize=16,
            color=(
                GREEN
                if row["accuracy_montants"] == 100
                else ORANGE
            ),
            weight="bold"
        )

        ax.text(
            0.07, y - 0.12,
            f"Cells: {row['cellules_correctes']}/"
            f"{row['cellules_testees']}",
            fontsize=10,
            color=MUTED
        )

    # ======================================
    # ACCOUNTING BALANCE
    # ======================================

    ax = panel(
        0.345, 0.075,
        0.30, 0.285,
        "Accounting balance"
    )

    for i, row in enumerate(balances):

        y = 0.65 - i * 0.27

        ax.text(
            0.07, y,
            row["exercice"],
            fontsize=11
        )

        ax.text(
            0.93, y,
            "PASS" if row["equilibre"] else "FAIL",
            ha="right",
            fontsize=12,
            color=GREEN if row["equilibre"] else RED,
            weight="bold"
        )

        value = row["total_actif"]

        ax.text(
            0.07, y - 0.12,
            (
                f"ACTIF: {value:,.2f}"
                if value is not None
                else "ACTIF: missing"
            ),
            fontsize=10,
            color=MUTED
        )

    # ======================================
    # ERROR ANALYSIS
    # ======================================

    ax = panel(
        0.66, 0.075,
        0.305, 0.285,
        "Error analysis"
    )

    total_errors = sum(errors.values())

    ax.text(
        0.07, 0.53,
        str(total_errors),
        fontsize=30,
        color=GREEN if total_errors == 0 else RED,
        weight="bold"
    )

    ax.text(
        0.07, 0.37,
        "Tested anomalies",
        fontsize=9,
        color=MUTED
    )

    ax.text(
        0.58, 0.53,
        str(len(metadata)),
        fontsize=30,
        color=ORANGE,
        weight="bold"
    )

    ax.text(
        0.58, 0.37,
        "Metadata absent",
        fontsize=9,
        color=MUTED
    )

    ax.text(
        0.07, 0.13,
        "Label transcription not fully tested",
        fontsize=9,
        color=MUTED
    )

    # ======================================
    # FOOTER
    # ======================================

    fig.text(
        0.035, 0.027,
        "ENSAM Casablanca | Transformers & LLMs | 2026",
        fontsize=9,
        color=MUTED
    )

    fig.text(
        0.965, 0.027,
        "Single-document evaluation | No additional inference",
        ha="right",
        fontsize=9,
        color=MUTED
    )

    # ======================================
    # EXPORT
    # ======================================

    fig.savefig(
        FIGURES / "dashboard_local.png",
        dpi=300,
        facecolor=BG
    )

    fig.savefig(
        FIGURES / "dashboard_local.pdf",
        facecolor=BG
    )

    # ======================================
    # FULLSCREEN DISPLAY
    # ======================================

    def keyboard(event):

        if event.key == "escape":
            plt.close(fig)

    fig.canvas.mpl_connect(
        "key_press_event",
        keyboard
    )

    try:

        manager = plt.get_current_fig_manager()

        manager.full_screen_toggle()

    except (AttributeError, NotImplementedError):

        pass

    print("\nDashboard generated successfully.")

    print("Press ESC to close.")

    plt.show()


# ==========================================
# 12. EXECUTION
# ==========================================

if __name__ == "__main__":

    main()
