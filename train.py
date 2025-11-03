import os
import json
import joblib
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# === 1. Charger les données ===
iris = load_iris()
X, y = iris.data, iris.target

# === 2. Split train/test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 3. Entraînement du modèle ===
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

# === 4. Évaluation ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# === 5. Sauvegarder le modèle ===
# Créer le dossier "models" s'il n'existe pas
os.makedirs("models", exist_ok=True)

# Sauvegarder le modèle entraîné
model_path = os.path.join("models", "iris_model.pkl")
joblib.dump(model, model_path)

# === 6. Sauvegarder les métriques ===
metrics = {
    "accuracy": accuracy,
    "n_estimators": 100,
    "test_size": len(X_test)
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# === 7. Afficher le résultat ===
print(f"✅ Modèle sauvegardé dans {model_path}")
print(f"✅ Accuracy: {accuracy:.4f}")
