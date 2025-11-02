from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# 🔄 Chargement des objets nécessaires
model = joblib.load("xgboost_voiture_model.joblib")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")
mean_encoded = joblib.load("mean_encoded_dict.joblib")
global_mean = joblib.load("global_mean.joblib")
feature_names = joblib.load("feature_names.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print("🔍 Données reçues par l'API :", data)

    if not data:
        return jsonify({'error': 'Données manquantes'}), 400

    try:
        # ✅ Construire "Marque et Modèle" à partir des champs séparés
        marque = data.get("marque", "").strip()
        modele = data.get("modele", "").strip()
        marque_modele = f"{marque} {modele}".strip()

        # ✅ Encodage
        marque_enc = mean_encoded.get(marque_modele, global_mean)
        print(f"💬 Marque et Modèle : {marque_modele}")
        print(f"💬 Marque encodée : {marque_enc}")

        # ✅ Validation de l’année de dédouanement
        annee_dedouane = data["vehicule_dedouane"]
        if not (0 <= annee_dedouane <= 2025):
            return jsonify({'error': 'Année de dédouanement invalide'}), 400

        # ✅ Construction du vecteur d'entrée
        input_dict = {
            "Kilométrage": data["kilometrage"],
            "Année": data["annee"],
            "Boite de vitesses": data["boite_vitesses"],
            "Carburant": data["carburant"],
            "Puissance fiscale": data["puissance_fiscale"],
            "Nombre de portes": data["nombre_portes"],
            "Première main": data["premiere_main"],
            "Véhicule dédouané": annee_dedouane,
            "Importé neuf": data["importe_neuf"],
            "Marque et Modèle": marque_enc
        }

        print("🔧 Données envoyées :", input_dict)
        print(f"🔧 Noms des caractéristiques du modèle : {feature_names}")

        X = np.array([[input_dict[col] for col in feature_names]])
        print(f"🔧 Vecteur d'entrée formaté : {X}")

        # 🧪 Standardisation
        X_scaled = scaler_X.transform(X)
        print(f"🧪 Données standardisées : {X_scaled}")

        # 🔮 Prédiction
        prediction = model.predict(X_scaled)
        print(f"📉 Prédiction standardisée : {prediction}")

        # 💰 Déstandardisation
        prix_estime_standard = prediction[0]
        prix_estime = scaler_y.inverse_transform([[prix_estime_standard]])[0][0]
        print(f"📉 Prix estimé déstandardisé : {prix_estime}")

        return jsonify({'prix_estime': round(prix_estime, 2)})

    except Exception as e:
        return jsonify({'error': f"Erreur lors de la prédiction : {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
