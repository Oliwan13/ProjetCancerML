#!/bin/bash

# Liste des dossiers/fichiers nécessaires
CHECKLIST=("models/scaler.pkl" "models/NN_type_trained_.pkl" "fonts/DejaVuSans.ttf" "server.py")

# Vérification
for ITEM in "${CHECKLIST[@]}"; do
  if [ ! -e "$ITEM" ]; then
    echo "❌ Manquant : $ITEM"
    exit 1
  else
    echo "✅ $ITEM OK"
  fi
done

# Activation de l'environnement virtuel
source env/bin/activate

# Lancement de Streamlit
echo "🚀 Lancement de l'application Streamlit..."
streamlit run server.py
