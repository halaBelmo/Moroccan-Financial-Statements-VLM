import json
from pathlib import Path
from jsonschema import Draft202012Validator

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "outputs" / "bilan_group_2_df6bceb3534c"

RESULTS_DIR = BASE_DIR / "evaluation" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

fichiers = {
    "ACTIF": OUTPUT_DIR / "actif_page_1.json",
    "PASSIF": OUTPUT_DIR / "passif_page_2.json"
}

resultats = []

for nom, fichier in fichiers.items():

    champs_montants = ["net_N", "net_N_1"]

    if nom == "ACTIF":
        champs_montants += [
            "brut",
            "amortissements_provisions"
        ]

    proprietes = {
        "source_page": {"type": "integer"},
        "source_row": {"type": "integer"},
        "poste": {"type": "string", "minLength": 1},
        "rubrique": {"type": ["string", "null"]},
        "type_ligne": {
            "enum": ["rubrique", "detail", "total"]
        }
    }

    for champ in champs_montants:
        proprietes[champ] = {
            "type": ["number", "null"]
        }

    schema = {
        "type": "object",
        "required": ["type_document", "lignes"],
        "properties": {
            "type_document": {"const": nom},
            "lignes": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": list(proprietes.keys()),
                    "properties": proprietes
                }
            }
        }
    }

    try:
        with open(fichier, encoding="utf-8") as f:
            data = json.load(f)

        validator = Draft202012Validator(schema)
        erreurs = list(validator.iter_errors(data))

        resultat = {
            "document": nom,
            "json_valide": True,
            "schema_valide": len(erreurs) == 0,
            "nombre_lignes": len(data.get("lignes", [])),
            "nombre_erreurs": len(erreurs),
            "erreurs": [e.message for e in erreurs]
        }

    except Exception as e:
        resultat = {
            "document": nom,
            "json_valide": False,
            "schema_valide": False,
            "nombre_erreurs": 1,
            "erreurs": [str(e)]
        }

    resultats.append(resultat)

    print(f"\n===== {nom} =====")

    for cle, valeur in resultat.items():
        print(f"{cle} : {valeur}")

# Sauvegarde des résultats du test
with open(
    RESULTS_DIR / "test_01_json.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(resultats, f, indent=4, ensure_ascii=False)