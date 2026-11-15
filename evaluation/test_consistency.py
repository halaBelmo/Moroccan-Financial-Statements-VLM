import json
from pathlib import Path
from decimal import Decimal

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs" / "bilan_group_2_df6bceb3534c"
RESULTS = BASE / "evaluation" / "results"

resultats = []

def montant(valeur):
    return Decimal(str(valeur)) if valeur is not None else Decimal("0")

def verifier(nom, controle, colonne, lignes, total):

    somme = sum(
        (montant(l.get(colonne)) for l in lignes),
        Decimal("0")
    )

    attendu = montant(total.get(colonne))
    ecart = abs(somme - attendu)

    resultat = {
        "document": nom,
        "controle": controle,
        "colonne": colonne,
        "somme_calculee": float(somme),
        "total_extrait": float(attendu),
        "ecart": float(ecart),
        "coherent": ecart == 0
    }

    resultats.append(resultat)


for nom, fichier in [
    ("ACTIF", "actif_page_1.json"),
    ("PASSIF", "passif_page_2.json")
]:

    with open(OUTPUT / fichier, encoding="utf-8") as f:
        data = json.load(f)

    lignes = data["lignes"]

    colonnes = ["net_N", "net_N_1"]

    if nom == "ACTIF":
        colonnes += ["brut", "amortissements_provisions"]

    # 1. Cohérence globale
    sous_totaux = [
        l for l in lignes
        if l["poste"].upper().split("(")[0].strip()
        in ["TOTAL I", "TOTAL II", "TOTAL III"]
    ]

    total_general = next(
        l for l in lignes
        if "TOTAL GENERAL" in l["poste"].upper()
    )

    assert len(sous_totaux) == 3, "Sous-totaux manquants"

    for colonne in colonnes:
        verifier(
            nom, "Cohérence globale",
            colonne, sous_totaux, total_general
        )

    # 2. Cohérence interne des rubriques
    for rubrique in lignes:

        if rubrique["type_ligne"] != "rubrique":
            continue

        code = rubrique["rubrique"]

        details = [
            l for l in lignes
            if l["type_ligne"] == "detail"
            and l["rubrique"] == code
        ]

        if not details:
            continue

        for colonne in colonnes:
            verifier(
                nom, rubrique["poste"],
                colonne, details, rubrique
            )

# Afficher les résultats
for r in resultats:

    statut = "OK" if r["coherent"] else "ERREUR"

    print(
        f"{r['document']} | "
        f"{r['controle']} | "
        f"{r['colonne']} | "
        f"{statut} | Ecart = {r['ecart']}"
    )

total = len(resultats)
reussis = sum(r["coherent"] for r in resultats)
taux = reussis / total * 100

print("\n===== RESULTATS =====")
print("Controles effectues :", total)
print("Controles reussis :", reussis)
print("Controles echoues :", total - reussis)
print(f"Taux de coherence : {taux:.2f}%")

RESULTS.mkdir(parents=True, exist_ok=True)

with open(
    RESULTS / "test_05_consistency.json",
    "w", encoding="utf-8"
) as f:
    json.dump({
        "controles": resultats,
        "total": total,
        "reussis": reussis,
        "taux_coherence": taux
    }, f, indent=4, ensure_ascii=False)