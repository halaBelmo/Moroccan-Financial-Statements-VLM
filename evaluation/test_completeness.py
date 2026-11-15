import json
from pathlib import Path
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs" / "bilan_group_2_df6bceb3534c"

RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Identifiants des lignes de référence
references = {
    "ACTIF": set(range(2, 53)),
    "PASSIF": set(range(1, 15))
              | set(range(19, 22))
              | set(range(26, 53))
}

fichiers = {
    "ACTIF": "actif_page_1.json",
    "PASSIF": "passif_page_2.json"
}

resultats = []

for nom, fichier in fichiers.items():

    with open(OUTPUT_DIR / fichier, encoding="utf-8") as f:
        data = json.load(f)

    attendus = references[nom]

    ids = [ligne["source_row"] for ligne in data["lignes"]]
    extraits = set(ids)

    TP = len(attendus & extraits)
    FP = len(extraits - attendus)
    FN = len(attendus - extraits)

    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    doublons = [
        i for i, n in Counter(ids).items() if n > 1
    ]

    resultat = {
        "document": nom,
        "lignes_attendues": len(attendus),
        "lignes_extraites": len(ids),
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "manquants": sorted(attendus - extraits),
        "supplementaires": sorted(extraits - attendus),
        "doublons": doublons,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1_score": f1 * 100
    }

    resultats.append(resultat)

    print(f"\n===== {nom} =====")

    for cle, valeur in resultat.items():
        print(f"{cle} : {valeur}")

with open(
    RESULTS_DIR / "test_02_completeness.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(resultats, f, indent=4, ensure_ascii=False)