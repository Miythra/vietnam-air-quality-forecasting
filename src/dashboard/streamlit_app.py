import streamlit as st
import os
import pandas as pd

st.set_page_config(page_title="Debug Mode", page_icon="🔧")

st.title("🕵️ Détective de Fichier")

# 1. Où est le script ?
current_dir = os.getcwd()
st.info(f"📂 **Le script s'exécute ici :** `{current_dir}`")

# 2. Recherche automatique du fichier 'aqi_data.csv'
target_file = "aqi_data.csv"
found_path = None

st.write(f"🔍 **Recherche de `{target_file}` dans tout le projet...**")

# On parcourt TOUS les dossiers et sous-dossiers pour le trouver
results = []
for root, dirs, files in os.walk("."):
    for file in files:
        if file == target_file:
            full_path = os.path.join(root, file)
            results.append(full_path)

# 3. Affichage du verdict
if len(results) > 0:
    st.success(f"✅ FICHIER TROUVÉ !")
    st.write("Voici les chemins trouvés (copie celui qui convient) :")
    for p in results:
        st.code(p, language="bash")
        
    # Test de lecture immédiat
    try:
        df = pd.read_csv(results[0])
        st.write(f"📊 **Test de lecture :** Réussi ! ({len(df)} lignes chargées)")
        st.dataframe(df.head(3))
    except Exception as e:
        st.error(f"Le fichier est là mais illisible : {e}")

else:
    st.error(f"❌ FICHIER INTROUVABLE.")
    st.warning("⚠️ Le fichier `aqi_data.csv` n'est PAS dans ce déploiement GitHub.")
    st.write("### Liste de tout ce qui existe ici :")
    # Affiche tout pour qu'on comprenne l'erreur
    all_files = []
    for root, dirs, files in os.walk("."):
        for name in files:
            all_files.append(os.path.join(root, name))
    st.code("\n".join(all_files[:50])) # On affiche les 50 premiers fichiers
