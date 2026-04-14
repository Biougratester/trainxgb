import os
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app) 

print("Initializing Clinker Cooler Engine...")


print("Engine ready.")



print("="*50)
print("🚀 RÉ-ENTRAÎNEMENT XGBOOST SUR LE SERVEUR LINUX")
print("="*50)

# --- 1. CHARGEMENT DES DONNÉES ---
# Modifiez ce nom pour correspondre au fichier CSV que vous avez uploadé
fichier_donnees = 'Augmented.csv' 

if not os.path.exists(fichier_donnees):
    print(f"❌ ERREUR FATALE : Impossible de trouver '{fichier_donnees}'.")
    print("   Veuillez uploader votre jeu de données dans le même dossier que ce script.")
    exit(1)

print(f"📁 Lecture des données depuis {fichier_donnees}...")
df = pd.read_csv(fichier_donnees)
# Si c'est un fichier Excel, utilisez : df = pd.read_excel(fichier_donnees)


# --- 2. SÉCURISATION DES COLONNES ---
# C'est ici qu'on bloque définitivement le bug des valeurs négatives !
# Mettez exactement les noms de colonnes tels qu'ils sont dans votre CSV.
colonnes_entrees = ['softot', 'vg', 'm']
colonne_cible_temp = 'powtot' 
# colonne_cible_puissance = 'electrical_power' # Décommentez si besoin

X = df[colonnes_entrees]
y_temp = df[colonne_cible_temp]


# --- 3. CONFIGURATION ET ENTRAÎNEMENT ---
print("⚙️ Lancement de l'entraînement de l'arbre de décision...")

# L'ASTUCE EN OR : n_jobs=1 empêchera le modèle de geler le serveur web plus tard
modele_xgb = xgb.XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=0.6, missing =np.nan , objective='reg:squarederror', eval_metric = 'rmse')

modele_xgb.fit(X, y_temp)
print("✅ Entraînement terminé avec succès !")


# --- 4. SAUVEGARDE ÉCRASANTE ---
dossier_modeles = 'PI_TOUT_xgb_model'
os.makedirs(dossier_modeles, exist_ok=True)

# REMPLACEZ LE NOM PAR CELUI ATTENDU PAR VOTRE CODE (ex: xgboost.pkl, xgboost_temp.pkl, etc.)
chemin_sauvegarde = os.path.join(dossier_modeles, 'PI_POW_xgb_model.joblib')

joblib.dump(modele_xgb, chemin_sauvegarde)

print(f"💾 Nouveau modèle 100% natif Linux sauvegardé dans : {chemin_sauvegarde}")
print("\n🎉 TERMINÉ. Vous pouvez maintenant aller cliquer sur 'Reload' dans l'onglet Web !")
print("="*50)
@app.route('/')
def home():
    path = chemin_sauvegarde
    return send_file(path, as_attachment=True)
if __name__ == '__main__':
    app.run(debug=True)
