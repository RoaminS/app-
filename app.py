import streamlit as st
import plotly.graph_objs as go
import numpy as np
import time
import os
import mne
from streamlit.components.v1 import html

# --- CONFIGURATION GÉNÉRALE ---
st.set_page_config(page_title="🧠 Visualiseur EEG 3D", layout="wide", page_icon="🧠")

st.markdown("""
<style>
    body {
        background-color: #0a0a0a;
        color: #ffffff;
        font-family: 'Orbitron', sans-serif;
    }
    .stButton>button {
        background-color: #1f1f2e;
        color: #ffffff;
        border-radius: 12px;
        border: none;
        transition: 0.3s;
        font-size: 18px;
    }
    .stButton>button:hover {
        background-color: #292945;
        transform: scale(1.05);
    }
    .block-container {
        padding-top: 2rem;
    }
    .neon-box {
        border: 2px solid #ff66cc;
        padding: 1rem;
        border-radius: 10px;
        background: rgba(255, 255, 255, 0.05);
        box-shadow: 0 0 10px #ff66cc;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# --- ANIMATION HUD & PARTICULES ---
html("""
<div style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:0;">
  <svg width="100%" height="100%">
    <defs>
      <radialGradient id="pulse" cx="50%" cy="50%" r="50%">
        <stop offset="0%" stop-color="#ff66cc" stop-opacity="0.8">
          <animate attributeName="stop-opacity" values="0.2;0.8;0.2" dur="3s" repeatCount="indefinite" />
        </stop>
        <stop offset="100%" stop-color="#0a0a0a" stop-opacity="0" />
      </radialGradient>
    </defs>
    <circle cx="50%" cy="50%" r="20%" fill="url(#pulse)" />
  </svg>
</div>
""", height=0)

# --- FONCTION POUR CHARGER EEG ---
def charger_eeg(fichier):
    if fichier.name.endswith(".edf"):
        raw = mne.io.read_raw_edf(fichier, preload=True)
    elif fichier.name.endswith(".csv"):
        data = np.loadtxt(fichier, delimiter=',')
        raw = mne.io.RawArray(data, mne.create_info(ch_names=[f"Ch{i}" for i in range(data.shape[0])], sfreq=256.0))
    elif fichier.name.endswith(".mat"):
        from scipy.io import loadmat
        mat = loadmat(fichier)
        key = list(mat.keys())[-1]
        data = mat[key]
        raw = mne.io.RawArray(data, mne.create_info(ch_names=[f"Ch{i}" for i in range(data.shape[0])], sfreq=256.0))
    else:
        st.error("Format non supporté. Formats acceptés : .edf, .csv, .mat")
        return None
    return raw

# --- AFFICHAGE 3D DU CERVEAU + RESEAU NEURONAL ---
def afficher_modele_threejs():
    try:
        with open("assets/3dbrain_embed.html", "r") as f:
            brain_html = f.read()
        html(brain_html, height=600, scrolling=False)
    except:
        st.warning("Fichier '3dbrain_embed.html' manquant dans le dossier assets.")

def afficher_reseau_neuronal():
    try:
        with open("assets/neural_overlay.html", "r") as f:
            neuron_html = f.read()
        html(neuron_html, height=600, scrolling=False)
    except:
        st.warning("Fichier 'neural_overlay.html' manquant dans le dossier assets.")

# --- INTERFACE PRINCIPALE ---
st.markdown("""
<h1 style='text-align: center; color: #ff66cc;'>🧠 Visualiseur EEG 3D</h1>
<p style='text-align: center;'>Explorez votre cerveau en temps réel ou avec des fichiers EEG, dans une interface immersive futuriste.</p>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h4 style='text-align:center;'>Cerveau 3D Interactif + Réseau Neuronal</h4>", unsafe_allow_html=True)
    afficher_modele_threejs()
    afficher_reseau_neuronal()
    st.markdown("""
    <div style='text-align: center;'>
        <button class="button">📂 Import EEG</button>
        <button class="button">🔴 Live EEG</button>
    </div>
    """, unsafe_allow_html=True)

# --- IMPORT EEG ---
fichier_eeg = st.file_uploader("Choisissez un fichier EEG", type=["edf", "csv", "mat"])
if fichier_eeg:
    raw = charger_eeg(fichier_eeg)
    if raw:
        st.success("Fichier EEG chargé avec succès.")
        fig = go.Figure()
        data, times = raw[:, :1000]
        for i, trace in enumerate(data):
            fig.add_trace(go.Scatter(y=trace, mode='lines', name=f"Ch{i}"))
        fig.update_layout(title='Signaux EEG (premières secondes)', template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        psds, freqs = mne.time_frequency.psd_welch(raw, fmin=1, fmax=50, n_fft=256)
        bande_dominante = freqs[np.argmax(psds.mean(axis=1))]
        if bande_dominante < 4:
            zone = "Temporal (Delta)"
        elif bande_dominante < 8:
            zone = "Frontal (Theta)"
        elif bande_dominante < 13:
            zone = "Occipital (Alpha)"
        elif bande_dominante < 30:
            zone = "Cortex moteur (Beta)"
        else:
            zone = "Fronto-pariétal (Gamma)"

        st.markdown(f"### 🔥 Activité dominante détectée : `{zone}`")

# --- MODE DÉMO ---
if st.button("🎬 Mode Démo"):
    fichier_demo = os.path.join("demo_data", "demo_eeg.edf")
    if os.path.exists(fichier_demo):
        raw = mne.io.read_raw_edf(fichier_demo, preload=True)
        st.success("Mode démo activé. Données EEG chargées.")
        st.write(raw.info)
    else:
        st.warning("Fichier démo non trouvé. Ajoutez-le dans le dossier `demo_data`.")

# --- INFOS ---
with st.expander("❔ Aide & Explications"):
    st.markdown("""
    - **📂 Import EEG** : Chargez un fichier EEG local (.edf, .csv, .mat).
    - **🔴 Live EEG** : Connexion à un casque EEG en temps réel (Bluetooth ou port série).
    - **Zones Cérébrales** : Les régions s'illuminent en fonction des ondes détectées (alpha, beta, gamma...)
    - **Graphique EEG** : Affiche l'évolution des signaux pour chaque canal.
    - **Mode Démo** : Utilisez un exemple intégré pour tester les fonctionnalités.
    """)
