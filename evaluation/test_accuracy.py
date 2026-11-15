import json
from pathlib import Path
from decimal import Decimal

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs" / "bilan_group_2_df6bceb3534c"
EVAL = BASE / "evaluation"

with open(EVAL / "ground_truth/amounts.json", encoding="utf-8") as f:
    ground_truth = json.load(f)

config = {
    "ACTIF": {
        "fichier": "actif_page_1.json",
        "colonnes": [
            "brut",
            "amortissements_provisions",
            "net_N",
            "net_N_1"
        ],
        "ids": list(range(2, 53))
    },
    "PASSIF": {
        "fichier": "passif_page_2.json",
        "colonnes": ["net_N", "net_N_1"],
        "ids": list(range(1, 15))
               + list(range(19, 22))
               + list(range(26, 53))
    }
}

resultats = []

for nom, cfg in config.items():

    with open(OUTPUT / cfg["fichier"], encoding="utf-8") as f:
        data = json.load(f)

    lignes = {
        ligne["source_row"]: ligne
        for ligne in data["lignes"]
    }

    total = 0
    corrects = 0
    montants_total = 0
    montants_corrects = 0
    erreurs = []

    for row_id in cfg["ids"]:

        reference = ground_truth[nom].get(
            str(row_id),
            [None] * len(cfg["colonnes"])
        )

        ligne = lignes.get(row_id, {})

        for colonne, attendu in zip(cfg["colonnes"], reference):

            extrait = ligne.get(colonne)
            total += 1

            if attendu is not None:
                montants_total += 1

            if attendu is None or extrait is None:
                correct = attendu is None and extrait is None
            else:
                correct = (
                    Decimal(str(attendu))
                    == Decimal(str(extrait))
                )

            correct = correct and colonne in ligne

            if correct:
                corrects += 1

                if attendu is not None:
                    montants_corrects += 1

            else:
                erreurs.append({
                    "source_row": row_id,
                    "poste": ligne.get("poste"),
                    "colonne": colonne,
                    "attendu": attendu,
                    "extrait": extrait
                })

    resultat = {
        "document": nom,
        "cellules_testees": total,
        "cellules_correctes": corrects,
        "cellules_incorrectes": total - corrects,
        "accuracy_globale": round(corrects / total * 100, 2),
        "montants_testes": montants_total,
        "montants_corrects": montants_corrects,
        "accuracy_montants": round(
            montants_corrects / montants_total * 100, 2
        ),
        "erreurs": erreurs
    }

    resultats.append(resultat)

    print(f"\n===== {nom} =====")

    for cle, valeur in resultat.items():
        print(f"{cle} : {valeur}")

(EVAL / "results").mkdir(exist_ok=True)

with open(
    EVAL / "results/test_03_accuracy.json",
    "w", encoding="utf-8"
) as f:
    json.dump(resultats, f, indent=4, ensure_ascii=False)