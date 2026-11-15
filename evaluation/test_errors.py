import json
from pathlib import Path
from collections import Counter

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs" / "bilan_group_2_df6bceb3534c"
RESULTS = BASE / "evaluation" / "results"

def charger(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

# Recuperer les resultats des tests precedents
validity = charger(RESULTS / "test_01_json.json")
completeness = charger(RESULTS / "test_02_completeness.json")
accuracy = charger(RESULTS / "test_03_accuracy.json")
balance = charger(RESULTS / "test_04_balance.json")
consistency = charger(RESULTS / "test_05_consistency.json")

# Classifier les erreurs detectees
erreurs = {
    "Erreurs de structure": sum(
        r["nombre_erreurs"] for r in validity
    ),
    "Postes manquants": sum(
        r["FN"] for r in completeness
    ),
    "Postes supplementaires": sum(
        r["FP"] for r in completeness
    ),
    "Doublons": sum(
        len(r["doublons"]) for r in completeness
    ),
    "Erreurs de montants": sum(
        r["cellules_incorrectes"] for r in accuracy
    ),
    "Desequilibres comptables": sum(
        not r["equilibre"] for r in balance["controles"]
    ),
    "Incoherences comptables": sum(
        not r["coherent"] for r in consistency["controles"]
    )
}

# Examiner les metadonnees
metadonnees_manquantes = []

for fichier in ["actif_page_1.json", "passif_page_2.json"]:

    data = charger(OUTPUT / fichier)

    for champ in ["entreprise", "date_cloture"]:
        if data.get(champ) is None:
            metadonnees_manquantes.append({
                "document": data["type_document"],
                "champ": champ
            })

# Afficher les resultats
print("\n===== ERROR ANALYSIS =====")

for categorie, nombre in erreurs.items():
    print(f"{categorie} : {nombre}")

print("\nTotal des anomalies detectees :", sum(erreurs.values()))

print("\n===== METADONNEES MANQUANTES =====")

for element in metadonnees_manquantes:
    print(element["document"], "|", element["champ"])

# Sauvegarder les resultats
rapport = {
    "erreurs": erreurs,
    "total_anomalies": sum(erreurs.values()),
    "metadonnees_manquantes": metadonnees_manquantes,
    "exactitude_textuelle": "A verifier manuellement"
}

with open(RESULTS / "test_06_errors.json",
          "w", encoding="utf-8") as f:
    json.dump(rapport, f, indent=4, ensure_ascii=False)