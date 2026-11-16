
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE = Path(__file__).resolve().parent

API_RESULTS = BASE / "evaluation" / "results"

LOCAL_RESULTS = BASE / "Local" / "evaluation" / "results"

OUTPUT = BASE / "comparison_results"
OUTPUT.mkdir(parents=True, exist_ok=True)

API_ESTIMATED_LATENCY = 40.0

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================

def load(folder, filename):
    path = folder / filename

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def pct(a, b):
    return 100 * a / b if b else 0


def format_time(seconds):

    if seconds is None:
        return "Unavailable"

    minutes, seconds = divmod(round(seconds), 60)

    if minutes == 0:
        return f"{seconds} s"

    return f"{minutes}m {seconds:02d}s"


# ==========================================
# 3. LOAD EVALUATION RESULTS
# ==========================================

def get_metrics(folder):

    validity = load(folder, "test_01_json.json")
    completeness = load(folder, "test_02_completeness.json")
    accuracy = load(folder, "test_03_accuracy.json")
    balance = load(folder, "test_04_balance.json")
    consistency = load(folder, "test_05_consistency.json")
    errors = load(folder, "test_06_errors.json")

    tp = sum(r["TP"] for r in completeness)
    fp = sum(r["FP"] for r in completeness)
    fn = sum(r["FN"] for r in completeness)

    precision = pct(tp, tp + fp)
    recall = pct(tp, tp + fn)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    correct = sum(
        r["montants_corrects"] for r in accuracy
    )

    amounts = sum(
        r["montants_testes"] for r in accuracy
    )

    cells_correct = sum(
        r["cellules_correctes"] for r in accuracy
    )

    cells_total = sum(
        r["cellules_testees"] for r in accuracy
    )

    latency = None
    latency_source = "Unavailable"

    latency_file = folder / "test_07_latency.json"

    if latency_file.exists():

        data = load(folder, "test_07_latency.json")

        value = data.get("total_secondes")

        if value is not None:
            latency = float(value)
            latency_source = "Recorded"

    return {

        "JSON Validity": pct(
            sum(
                r["json_valide"] and r["schema_valide"]
                for r in validity
            ),
            len(validity)
        ),

        "Completeness": recall,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,

        "Amount Accuracy": pct(correct, amounts),

        "Cell Accuracy": pct(
            cells_correct, cells_total
        ),

        "Accounting Balance": balance["taux_equilibre"],

        "Financial Consistency": consistency[
            "taux_coherence"
        ],

        "rows": tp,
        "expected_rows": tp + fn,

        "amounts_correct": correct,
        "amounts_total": amounts,

        "cells_correct": cells_correct,
        "cells_total": cells_total,

        "controls_passed": consistency["reussis"],
        "controls_total": consistency["total"],

        "balance_passed": sum(
            r["equilibre"]
            for r in balance["controles"]
        ),

        "balance_total": len(balance["controles"]),

        "errors": errors["total_anomalies"],

        "metadata": len(
            errors["metadonnees_manquantes"]
        ),

        "latency": latency,
        "latency_source": latency_source
    }


api = get_metrics(API_RESULTS)
local = get_metrics(LOCAL_RESULTS)

# Use an estimate when no API benchmark exists
if api["latency"] is None:

    api["latency"] = API_ESTIMATED_LATENCY

    api["latency_source"] = "Estimated"

# ==========================================
# 4. LATENCY COMPARISON
# ==========================================

speedup = None
time_reduction = None

if (
    api["latency"] is not None
    and local["latency"] is not None
    and api["latency"] > 0
    and local["latency"] > 0
):

    speedup = local["latency"] / api["latency"]

    time_reduction = (
        1 - api["latency"] / local["latency"]
    ) * 100

estimated = (
    api["latency_source"] == "Estimated"
    or local["latency_source"] == "Estimated"
)

# ==========================================
# 5. COMPARATIVE METRICS
# ==========================================

METRICS = [
    "JSON Validity",
    "Completeness",
    "Precision",
    "Recall",
    "F1-score",
    "Amount Accuracy",
    "Cell Accuracy",
    "Accounting Balance",
    "Financial Consistency"
]

comparison = []

print("\n========== API VS LOCAL ==========\n")

for metric in METRICS:

    comparison.append({
        "metric": metric,
        "api": api[metric],
        "local": local[metric],
        "difference": local[metric] - api[metric]
    })

    print(
        f"{metric:25s}"
        f" API: {api[metric]:.2f}%"
        f" | LOCAL: {local[metric]:.2f}%"
    )

print(
    "\nAPI latency:",
    format_time(api["latency"]),
    api["latency_source"]
)

print(
    "Local latency:",
    format_time(local["latency"]),
    local["latency_source"]
)

if speedup is not None:

    prefix = "Estimated" if estimated else "Observed"

    print(f"{prefix} speedup: {speedup:.2f}x")

    print(
        f"{prefix} time reduction: "
        f"{time_reduction:.2f}%"
    )

# ==========================================
# 6. SAVE COMPARISON REPORT
# ==========================================

with open(
    OUTPUT / "comparison.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "api": api,
            "local": local,
            "comparison": comparison,
            "speedup": speedup,
            "time_reduction_percentage": time_reduction,
            "latency_comparison_estimated": estimated,
            "scope": "Single financial statement",
            "text_accuracy": "Not fully evaluated"
        },
        f,
        indent=4,
        ensure_ascii=False
    )

# ==========================================
# 7. DESIGN SYSTEM
# ==========================================

BG = "#0B1220"
CARD = "#152136"
BORDER = "#283952"

TEXT = "#F1F5F9"
MUTED = "#94A3B8"

BLUE = "#38BDF8"
PURPLE = "#A78BFA"
GREEN = "#34D399"
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

# ==========================================
# 8. GRAPHICAL COMPONENTS
# ==========================================

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
            boxstyle="round,pad=0,rounding_size=0.025",
            facecolor=CARD,
            edgecolor=BORDER,
            linewidth=1
        )
    )

    if title:

        ax.text(
            0.05, 0.90,
            title,
            fontsize=14,
            weight="bold",
            va="center"
        )

    return ax


def progress(ax, x, y, width, value, color):

    ratio = max(0, min(value / 100, 1))

    ax.plot(
        [x, x + width],
        [y, y],
        color=BORDER,
        linewidth=4,
        solid_capstyle="round"
    )

    if ratio > 0:

        ax.plot(
            [x, x + width * ratio],
            [y, y],
            color=color,
            linewidth=4,
            solid_capstyle="round"
        )


def kpi(x, title, value, subtitle, color):

    ax = panel(x, 0.755, 0.22, 0.125)

    ax.text(
        0.07, 0.76,
        title,
        fontsize=10,
        color=MUTED
    )

    ax.text(
        0.07, 0.36,
        value,
        fontsize=26,
        color=color,
        weight="bold"
    )

    ax.text(
        0.07, 0.12,
        subtitle,
        fontsize=9,
        color=MUTED
    )


# ==========================================
# 9. HEADER
# ==========================================

fig.text(
    0.035, 0.945,
    "VLM BENCHMARK / API VS LOCAL",
    fontsize=24,
    weight="bold"
)

fig.text(
    0.035, 0.910,
    "Financial Document Intelligence | TALIANI AUTO",
    fontsize=11,
    color=MUTED
)

fig.text(
    0.965, 0.945,
    "COMPARATIVE EVALUATION",
    fontsize=10,
    color=BLUE,
    weight="bold",
    ha="right"
)

# ==========================================
# 10. KPI CARDS
# ==========================================

api_latency_label = format_time(api["latency"])

if api["latency_source"] == "Estimated":
    api_latency_label = "~" + api_latency_label

kpi(
    0.035,
    "GROQ API LATENCY",
    api_latency_label,
    api["latency_source"] + " inference time",
    BLUE
)

kpi(
    0.272,
    "LOCAL VLM LATENCY",
    format_time(local["latency"]),
    local["latency_source"] + " inference time",
    PURPLE
)

kpi(
    0.509,
    "ESTIMATED SPEEDUP" if estimated else "SPEEDUP",
    f"{speedup:.2f}x" if speedup is not None else "N/A",
    "Local time / API time",
    GREEN
)

kpi(
    0.746,
    "EST. TIME REDUCTION" if estimated else "TIME REDUCTION",
    f"{time_reduction:.2f}%" if time_reduction is not None else "N/A",
    "Relative inference duration",
    ORANGE
)

# ==========================================
# 11. QUALITY BENCHMARK
# ==========================================

ax = panel(
    0.035, 0.075,
    0.455, 0.65,
    "Quality benchmark"
)

ax.scatter(0.62, 0.83, color=BLUE, s=45)

ax.text(
    0.65, 0.83,
    "API",
    fontsize=10,
    va="center"
)

ax.scatter(0.79, 0.83, color=PURPLE, s=45)

ax.text(
    0.82, 0.83,
    "Local",
    fontsize=10,
    va="center"
)

for i, metric in enumerate(METRICS):

    y = 0.76 - i * 0.083

    ax.text(
        0.05, y,
        metric,
        fontsize=10,
        va="center"
    )

    for j, (data, color) in enumerate([
        (api, BLUE),
        (local, PURPLE)
    ]):

        yy = y - 0.028 - j * 0.023

        value = data[metric]

        progress(
            ax, 0.06, yy, 0.74,
            value, color
        )

        ax.text(
            0.95, yy,
            f"{value:.1f}%",
            fontsize=8,
            color=color,
            ha="right",
            va="center"
        )

# ==========================================
# 12. EXTRACTION RESULTS
# ==========================================

ax = panel(
    0.51, 0.405,
    0.455, 0.32,
    "Extraction results"
)

ax.text(
    0.67, 0.76,
    "API",
    color=BLUE,
    fontsize=10,
    weight="bold",
    ha="center"
)

ax.text(
    0.88, 0.76,
    "LOCAL",
    color=PURPLE,
    fontsize=10,
    weight="bold",
    ha="center"
)

rows = [
    ("Accounting rows", "rows", "expected_rows"),
    ("Financial amounts", "amounts_correct", "amounts_total"),
    ("Financial cells", "cells_correct", "cells_total"),
    ("Consistency checks", "controls_passed", "controls_total"),
    ("Balance checks", "balance_passed", "balance_total")
]

for i, (label, key, total_key) in enumerate(rows):

    y = 0.62 - i * 0.125

    ax.text(
        0.05, y,
        label,
        fontsize=11,
        va="center"
    )

    for x, data, color in [
        (0.67, api, BLUE),
        (0.88, local, PURPLE)
    ]:

        ax.text(
            x, y,
            f"{data[key]}/{data[total_key]}",
            fontsize=12,
            weight="bold",
            color=color,
            ha="center",
            va="center"
        )

# ==========================================
# 13. TECHNICAL COMPARISON
# ==========================================

ax = panel(
    0.51, 0.075,
    0.285, 0.30,
    "Technical comparison"
)

ax.text(
    0.53, 0.75,
    "API",
    fontsize=9,
    color=BLUE,
    weight="bold",
    ha="center"
)

ax.text(
    0.82, 0.75,
    "LOCAL",
    fontsize=9,
    color=PURPLE,
    weight="bold",
    ha="center"
)

technical_rows = [
    ("Parameters", "27B", "7B"),
    ("Platform", "GroqCloud", "Kaggle"),
    ("Resolution", "200 DPI", "300 DPI"),
    ("Segments", "11 blocks", "28 groups"),
    ("Quantization", "Unknown", "4-bit NF4")
]

for i, (label, api_value, local_value) in enumerate(
    technical_rows
):

    y = 0.61 - i * 0.12

    ax.text(
        0.06, y,
        label,
        fontsize=9
    )

    ax.text(
        0.53, y,
        api_value,
        fontsize=9,
        ha="center",
        color=BLUE
    )

    ax.text(
        0.82, y,
        local_value,
        fontsize=9,
        ha="center",
        color=PURPLE
    )

# ==========================================
# 14. ERROR ANALYSIS
# ==========================================

ax = panel(
    0.81, 0.075,
    0.155, 0.30,
    "Error analysis"
)

ax.text(
    0.53, 0.76,
    "API",
    fontsize=9,
    color=BLUE,
    ha="center"
)

ax.text(
    0.82, 0.76,
    "Local",
    fontsize=9,
    color=PURPLE,
    ha="center"
)

for i, (label, key) in enumerate([
    ("Errors", "errors"),
    ("Metadata", "metadata")
]):

    y = 0.56 - i * 0.27

    ax.text(
        0.07, y,
        label,
        fontsize=9,
        va="center"
    )

    for x, data in [
        (0.53, api),
        (0.82, local)
    ]:

        value = data[key]

        ax.text(
            x, y,
            str(value),
            fontsize=21,
            weight="bold",
            ha="center",
            va="center",
            color=GREEN if value == 0 else ORANGE
        )

ax.text(
    0.07, 0.07,
    "Text accuracy not tested fully",
    fontsize=7,
    color=MUTED
)

# ==========================================
# 15. FOOTER
# ==========================================

fig.text(
    0.035, 0.025,
    "ENSAM Casablanca | Transformers & LLMs | 2026",
    fontsize=9,
    color=MUTED
)

fig.text(
    0.965, 0.025,
    (
        "Single-document benchmark | API latency estimated"
        if estimated else
        "Single-document benchmark | Different pipelines"
    ),
    fontsize=8,
    color=MUTED,
    ha="right"
)

# ==========================================
# 16. EXPORT DASHBOARD
# ==========================================

fig.savefig(
    OUTPUT / "dashboard_comparison.png",
    dpi=300,
    facecolor=BG
)

fig.savefig(
    OUTPUT / "dashboard_comparison.pdf",
    facecolor=BG
)

# ==========================================
# 17. RESPONSIVE FULLSCREEN
# ==========================================

original_size = fig.get_size_inches().copy()

original_positions = [
    ax.get_position().frozen()
    for ax in fig.axes
]

original_fonts = [
    (text, text.get_fontsize())
    for text in fig.findobj(match=plt.Text)
]

original_text_positions = [
    (text, text.get_position())
    for text in fig.texts
]


def fit_screen(event=None):

    w, h = fig.get_size_inches()

    if w <= 0 or h <= 0:
        return

    desired_ratio = 16 / 9
    actual_ratio = w / h

    sx = min(1, desired_ratio / actual_ratio)
    sy = min(1, actual_ratio / desired_ratio)

    ox = (1 - sx) / 2
    oy = (1 - sy) / 2

    for ax, pos in zip(fig.axes, original_positions):

        ax.set_position([
            ox + pos.x0 * sx,
            oy + pos.y0 * sy,
            pos.width * sx,
            pos.height * sy
        ])

    for text, (x, y) in original_text_positions:

        text.set_position((
            ox + x * sx,
            oy + y * sy
        ))

    font_scale = min(
        w / original_size[0],
        h / original_size[1]
    )

    for text, size in original_fonts:

        text.set_fontsize(size * font_scale)

    fig.canvas.draw_idle()


def keyboard(event):

    if event.key == "escape":
        plt.close(fig)


fig.canvas.mpl_connect(
    "resize_event",
    fit_screen
)

fig.canvas.mpl_connect(
    "key_press_event",
    keyboard
)

try:

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()

except (AttributeError, NotImplementedError):

    pass

fit_screen()

print("\nComparison dashboard generated successfully.")
print("Saved in:", OUTPUT)
print("Press ESC to close.")

plt.show()
