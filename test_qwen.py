
import os
from dotenv import load_dotenv
from groq import Groq

# Charger les variables du fichier .env
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("Erreur : clé API introuvable.")
    exit()

# Connexion à Groq
client = Groq(api_key=api_key)

# Envoyer une question à Qwen
try:
    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": "Bonjour ! Présente-toi en français."
            }
        ],
        max_completion_tokens=500
    )

    print("Réponse de Qwen :")
    print(response.choices[0].message.content)

except Exception as e:
    print("Erreur :", e)


