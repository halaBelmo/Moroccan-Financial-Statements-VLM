import json
from pathlib import Path
from decimal import Decimal

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "outputs" / "bilan_group_2_df6bceb3534c"
RESULTS = BASE / "evaluation" / "results"

# Charger les fichiers JSON
with open(OUTPUT / "actif_page_1.json", encoding="utf-8") as f:
    actif = json.load(f)

with open(OUTPUT / "passif_page_2.json", encoding="utf-8") as f:
    passif = json.load(f)

# Extraire le total général
def trouver_total(data, colonne):
    for ligne in data["lignes"]:
        if "TOTAL GENERAL" in ligne["poste"].upper():
            valeur = ligne.get(colonne)

            if valeur is not None:
                return Decimal(str(valeur))

    raise ValueError(f"Total introuvable : {colonne}")

resultats = []

# Vérifier l'équilibre comptable
for exercice in ["net_N", "net_N_1"]:

    total_actif = trouver_total(actif, exercice)
    total_passif = trouver_total(passif, exercice)

    ecart = abs(total_actif - total_passif)
    equilibre = ecart == 0

    resultat = {
        "exercice": exercice,
        "total_actif": float(total_actif),
        "total_passif": float(total_passif),
        "ecart": float(ecart),
        "equilibre": equilibre
    }

    resultats.append(resultat)

    print(f"\n===== {exercice} =====")

    for cle, valeur in resultat.items():
        print(f"{cle} : {valeur}")

# Calculer le taux de réussite
reussis = sum(r["equilibre"] for r in resultats)
accuracy = reussis / len(resultats) * 100

print(f"\nControles reussis : {reussis}/{len(resultats)}")
print(f"Taux d'equilibre : {accuracy:.2f}%")

# Sauvegarder les résultats
RESULTS.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "test_04_balance.json", "w",
          encoding="utf-8") as f:
    json.dump({
        "controles": resultats,
        "taux_equilibre": accuracy
    }, f, indent=4, ensure_ascii=False)