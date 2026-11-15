
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIGURES = BASE / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)

def load(name):
    with open(RESULTS / name, encoding="utf-8") as f:
        return json.load(f)

# ==========================================
# 2. CHARGEMENT DES RESULTATS
# ==========================================

validity = load("test_01_json.json")
completeness = load("test_02_completeness.json")
accuracy = load("test_03_accuracy.json")
balance = load("test_04_balance.json")
consistency = load("test_05_consistency.json")
errors = load("test_06_errors.json")
comparison = load("test_07_comparison.json")

# ==========================================
# 3. CALCUL DES METRIQUES
# ==========================================

tp = sum(r["TP"] for r in completeness)
fp = sum(r["FP"] for r in completeness)
fn = sum(r["FN"] for r in completeness)

precision = 100 * tp / (tp + fp) if tp + fp else 0
recall = 100 * tp / (tp + fn) if tp + fn else 0

f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0
)

correct = sum(r["montants_corrects"] for r in accuracy)
total = sum(r["montants_testes"] for r in accuracy)

amount_rate = 100 * correct / total if total else 0

cells_correct = sum(r["cellules_correctes"] for r in accuracy)
cells_total = sum(r["cellules_testees"] for r in accuracy)

json_rate = 100 * sum(
    r["json_valide"] and r["schema_valide"]
    for r in validity
) / len(validity)

metadata_missing = len(errors["metadonnees_manquantes"])

actif = next(
    r["TP"] for r in completeness
    if r["document"] == "ACTIF"
)

passif = next(
    r["TP"] for r in completeness
    if r["document"] == "PASSIF"
)

# ==========================================
# 4. DESIGN SYSTEM
# ==========================================

BG = "#090F1D"
CARD = "#131D30"
BORDER = "#26354C"

TEXT = "#F1F5F9"
MUTED = "#94A3B8"

BLUE = "#38BDF8"
GREEN = "#34D399"
PURPLE = "#A78BFA"
ORANGE = "#FBBF24"
RED = "#FB7185"
TEAL = "#2DD4BF"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "toolbar": "None"
})

# Format fixe 16:9
fig = plt.figure(figsize=(16, 9), facecolor=BG)

# ==========================================
# 5. COMPOSANTS GRAPHIQUES
# ==========================================

def panel(x, y, w, h):

    ax = fig.add_axes([x, y, w, h])

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_navigate(False)

    ax.add_patch(
        FancyBboxPatch(
            (0.002, 0.002),
            0.996, 0.996,
            boxstyle="round,pad=0,rounding_size=0.025",
            transform=ax.transAxes,
            facecolor=CARD,
            edgecolor=BORDER,
            linewidth=1
        )
    )

    return ax


def progress_bar(ax, x, y, width, height, value, color):

    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0,rounding_size=0.009",
            facecolor=BORDER,
            linewidth=0
        )
    )

    if value > 0:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width * min(value / 100, 1),
                height,
                boxstyle="round,pad=0,rounding_size=0.009",
                facecolor=color,
                linewidth=0
            )
        )


# ==========================================
# 6. HEADER
# ==========================================

fig.text(
    0.035, 0.947,
    "VLM PERFORMANCE ANALYTICS",
    fontsize=23,
    fontweight="bold",
    color=TEXT
)

fig.text(
    0.035, 0.911,
    "Qwen  /  GroqCloud  /  OpenCV  /  TALIANI AUTO",
    fontsize=11,
    color=MUTED
)

fig.text(
    0.965, 0.947,
    "MODEL EVALUATION",
    ha="right",
    fontsize=11,
    fontweight="bold",
    color=BLUE
)

fig.text(
    0.965, 0.911,
    "Financial Document Intelligence",
    ha="right",
    fontsize=9,
    color=MUTED
)

# ==========================================
# 7. KPI CARDS
# ==========================================

def kpi(x, title, value, subtitle, color):

    ax = panel(x, 0.755, 0.22, 0.125)

    ax.text(
        0.075, 0.76,
        title,
        fontsize=10,
        color=MUTED
    )

    ax.text(
        0.075, 0.36,
        value,
        fontsize=27,
        fontweight="bold",
        color=color
    )

    ax.text(
        0.075, 0.12,
        subtitle,
        fontsize=9,
        color=MUTED
    )


kpi(
    0.035,
    "COMPLETENESS",
    f"{recall:.0f}%",
    f"{tp}/{tp + fn} accounting lines",
    BLUE
)

kpi(
    0.272,
    "AMOUNT ACCURACY",
    f"{amount_rate:.0f}%",
    f"{correct}/{total} correct amounts",
    GREEN
)

kpi(
    0.509,
    "CONSISTENCY",
    f"{consistency['taux_coherence']:.0f}%",
    f"{consistency['reussis']}/{consistency['total']} checks",
    PURPLE
)

kpi(
    0.746,
    "ACCOUNTING BALANCE",
    f"{balance['taux_equilibre']:.0f}%",
    "ACTIF = PASSIF",
    TEAL
)

# ==========================================
# 8. EXTRACTION METHODS
# ==========================================

ax = panel(0.035, 0.405, 0.455, 0.32)

ax.text(
    0.05, 0.89,
    "Extraction strategies",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    0.05, 0.79,
    "Numerical coverage by method",
    fontsize=9,
    color=MUTED
)

colors = [BLUE, RED, PURPLE, GREEN]

for i, row in enumerate(comparison):

    y = 0.65 - i * 0.165

    score = float(row["Couverture_globale"])

    ax.text(
        0.05, y,
        row["Methode"],
        fontsize=10
    )

    ax.text(
        0.95, y,
        f"{score:.2f}%",
        ha="right",
        fontsize=10,
        fontweight="bold",
        color=colors[i % len(colors)]
    )

    progress_bar(
        ax,
        0.05,
        y - 0.085,
        0.90,
        0.045,
        score,
        colors[i % len(colors)]
    )

# ==========================================
# 9. EXTRACTION COVERAGE - DONUT
# ==========================================

ax = panel(0.51, 0.405, 0.455, 0.32)

ax.text(
    0.05, 0.89,
    "Extraction coverage",
    fontsize=14,
    fontweight="bold"
)

donut = fig.add_axes([
    0.535, 0.425, 0.225, 0.235
])

donut.set_facecolor(CARD)
donut.set_navigate(False)

donut.pie(
    [actif, passif, fn] if tp + fn else [1],
    colors=[BLUE, PURPLE, BORDER],
    startangle=90,
    counterclock=False,
    wedgeprops={
        "width": 0.23,
        "edgecolor": CARD
    }
)

donut.text(
    0, 0.10,
    f"{recall:.0f}%",
    ha="center",
    va="center",
    fontsize=27,
    fontweight="bold",
    color=TEXT
)

donut.text(
    0, -0.23,
    "Coverage",
    ha="center",
    fontsize=9,
    color=MUTED
)

for y, label, value, color in [
    (0.61, "ACTIF", actif, BLUE),
    (0.38, "PASSIF", passif, PURPLE),
    (0.15, "Missing", fn, MUTED)
]:

    ax.scatter(
        0.59, y,
        s=65,
        color=color
    )

    ax.text(
        0.64, y,
        label,
        fontsize=10,
        va="center"
    )

    ax.text(
        0.93, y,
        str(value),
        fontsize=16,
        fontweight="bold",
        ha="right",
        va="center",
        color=color
    )

# ==========================================
# 10. MODEL EVALUATION
# ==========================================

ax = panel(0.035, 0.075, 0.295, 0.30)

ax.text(
    0.07, 0.88,
    "Model evaluation",
    fontsize=13,
    fontweight="bold"
)

metrics = [
    ("JSON Validity", json_rate),
    ("Precision", precision),
    ("Recall", recall),
    ("F1-score", f1)
]

for i, (name, score) in enumerate(metrics):

    y = 0.70 - i * 0.17

    ax.text(
        0.07, y,
        name,
        fontsize=10
    )

    ax.text(
        0.93, y,
        f"{score:.1f}%",
        fontsize=10,
        color=GREEN,
        ha="right",
        fontweight="bold"
    )

    progress_bar(
        ax,
        0.07,
        y - 0.075,
        0.86,
        0.035,
        score,
        GREEN
    )

# ==========================================
# 11. FINANCIAL AMOUNTS
# ==========================================

ax = panel(0.345, 0.075, 0.30, 0.30)

ax.text(
    0.06, 0.88,
    "Financial amount accuracy",
    fontsize=13,
    fontweight="bold"
)

chart = fig.add_axes([
    0.38, 0.125, 0.23, 0.175
])

chart.set_facecolor(CARD)
chart.set_navigate(False)

labels = [r["document"] for r in accuracy]
expected = [r["montants_testes"] for r in accuracy]
extracted = [r["montants_corrects"] for r in accuracy]

positions = list(range(len(labels)))
width = 0.29

b1 = chart.bar(
    [x - width / 2 for x in positions],
    expected,
    width=width,
    color=BLUE,
    label="Expected"
)

b2 = chart.bar(
    [x + width / 2 for x in positions],
    extracted,
    width=width,
    color=GREEN,
    label="Correct"
)

chart.bar_label(
    b1, padding=3,
    fontsize=9,
    color=TEXT
)

chart.bar_label(
    b2, padding=3,
    fontsize=9,
    color=TEXT
)

chart.set_xticks(positions)
chart.set_xticklabels(labels, fontsize=9)

chart.set_ylim(
    0, max(expected + extracted + [1]) * 1.3
)

chart.grid(
    axis="y",
    color=BORDER,
    alpha=0.4
)

chart.set_axisbelow(True)

for spine in chart.spines.values():
    spine.set_visible(False)

chart.tick_params(length=0)

chart.legend(
    loc="upper right",
    frameon=False,
    fontsize=8,
    labelcolor=MUTED
)

# ==========================================
# 12. ERROR ANALYSIS
# ==========================================

ax = panel(0.66, 0.075, 0.305, 0.30)

ax.text(
    0.07, 0.88,
    "Error analysis",
    fontsize=13,
    fontweight="bold"
)

ax.text(
    0.08, 0.57,
    str(errors["total_anomalies"]),
    fontsize=32,
    fontweight="bold",
    color=GREEN
)

ax.text(
    0.08, 0.42,
    "Tested anomalies",
    fontsize=9,
    color=MUTED
)

ax.text(
    0.59, 0.57,
    str(metadata_missing),
    fontsize=32,
    fontweight="bold",
    color=ORANGE
)

ax.text(
    0.59, 0.42,
    "Missing metadata",
    fontsize=9,
    color=MUTED
)

ax.plot(
    [0.07, 0.93],
    [0.32, 0.32],
    color=BORDER,
    linewidth=1
)

ax.text(
    0.07, 0.19,
    "Financial cells validated",
    fontsize=9,
    color=MUTED
)

ax.text(
    0.93, 0.18,
    f"{cells_correct}/{cells_total}",
    fontsize=13,
    ha="right",
    fontweight="bold",
    color=GREEN
)

# ==========================================
# 13. FOOTER
# ==========================================

fig.text(
    0.035, 0.027,
    "ENSAM Casablanca  |  Transformers & LLMs  |  2026",
    fontsize=9,
    color=MUTED
)

fig.text(
    0.965, 0.027,
    "Single-document evaluation | Text accuracy not fully tested",
    ha="right",
    fontsize=8,
    color=MUTED
)

# ==========================================
# 14. EXPORT
# ==========================================

fig.savefig(
    FIGURES / "dashboard_final.png",
    dpi=300,
    facecolor=BG
)

fig.savefig(
    FIGURES / "dashboard_final.pdf",
    facecolor=BG
)

# ==========================================
# 15. FIXED FULLSCREEN DISPLAY
# ==========================================

# Desactiver le zoom et la navigation
for ax in fig.axes:
    ax.set_navigate(False)

# Ajuster proportionnellement le dashboard a la fenetre
original_positions = [
    ax.get_position().frozen()
    for ax in fig.axes
]

original_size = fig.get_size_inches().copy()
aspect_ratio = 16 / 9

original_fonts = [
    (text, text.get_fontsize())
    for text in fig.findobj(match=plt.Text)
]

original_figure_texts = [
    (text, text.get_position())
    for text in fig.texts
]

def fit_to_screen(event=None):

    w, h = fig.get_size_inches()

    scale_x = min(1, aspect_ratio / (w / h))
    scale_y = min(1, (w / h) / aspect_ratio)

    offset_x = (1 - scale_x) / 2
    offset_y = (1 - scale_y) / 2

    for ax, pos in zip(fig.axes, original_positions):

        ax.set_position([
            offset_x + pos.x0 * scale_x,
            offset_y + pos.y0 * scale_y,
            pos.width * scale_x,
            pos.height * scale_y
        ])

    for text, (x, y) in original_figure_texts:
        text.set_position((
            offset_x + x * scale_x,
            offset_y + y * scale_y
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
    fit_to_screen
)

fig.canvas.mpl_connect(
    "key_press_event",
    keyboard
)

# Plein ecran
manager = plt.get_current_fig_manager()

try:
    manager.full_screen_toggle()
except Exception:
    pass

fit_to_screen()

print("Dashboard generated successfully.")
print("Press ESC to close the dashboard.")
print("Saved in:", FIGURES)

plt.show()
