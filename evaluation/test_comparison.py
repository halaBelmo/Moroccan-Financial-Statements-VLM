import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

RESULTS = BASE / "evaluation" / "results"
FIGURES = BASE / "evaluation" / "figures"
OUTPUT = BASE / "outputs" / "bilan_group_2_df6bceb3534c"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

# Recuperer les resultats finaux
with open(OUTPUT / "actif_page_1.json", encoding="utf-8") as f:
    actif = json.load(f)

with open(OUTPUT / "passif_page_2.json", encoding="utf-8") as f:
    passif = json.load(f)

# Resultats des experimentations
df = pd.DataFrame({
    "Methode": [
        "Extraction initiale",
        "Decoupage simple",
        "Guided Extraction",
        "OpenCV + VLM"
    ],
    "ACTIF": [
        44, 28, 51, len(actif["lignes"])
    ],
    "PASSIF": [
        38, 23, 44, len(passif["lignes"])
    ]
})

df["Total"] = df["ACTIF"] + df["PASSIF"]

df["Couverture_ACTIF"] = df["ACTIF"] / 51 * 100
df["Couverture_PASSIF"] = df["PASSIF"] / 44 * 100
df["Couverture_globale"] = df["Total"] / 95 * 100

df = df.round(2)

print("\n===== COMPARATIVE EVALUATION =====")
print(df.to_string(index=False))

# Sauvegarder les resultats
df.to_csv(
    RESULTS / "test_07_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

df.to_json(
    RESULTS / "test_07_comparison.json",
    orient="records",
    indent=4
)

# Diagramme 1 : Nombre de postes extraits
x = np.arange(len(df))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(
    x - width/2, df["ACTIF"], width,
    label="ACTIF", color="#2563eb"
)

b2 = ax.bar(
    x + width/2, df["PASSIF"], width,
    label="PASSIF", color="#f59e0b"
)

ax.bar_label(b1, padding=3)
ax.bar_label(b2, padding=3)

ax.set_xticks(x)
ax.set_xticklabels(df["Methode"])
ax.set_ylabel("Nombre de postes extraits")
ax.set_title("Comparaison des strategies d'extraction")
ax.set_ylim(0, 60)
ax.legend()
ax.grid(axis="y", alpha=0.15)

plt.tight_layout()
plt.savefig(
    FIGURES / "comparison_methods.png",
    dpi=300
)
plt.show()

# Diagramme 2 : Couverture globale
fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    df["Methode"],
    df["Couverture_globale"],
    color=["#3b82f6", "#ef4444", "#14b8a6", "#16a34a"]
)

ax.bar_label(bars, fmt="%.2f%%", padding=5)
ax.set_xlim(0, 115)
ax.set_xlabel("Couverture (%)")
ax.set_title("Evolution de la couverture des postes")
ax.axvline(100, linestyle="--", color="gray")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(
    FIGURES / "coverage_comparison.png",
    dpi=300
)
plt.show()