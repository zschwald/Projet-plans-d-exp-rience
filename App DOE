import numpy as np  # calculs numériques
import pandas as pd  # manipulation de données
import streamlit as st  # interface web
import plotly.express as px  # graphiques simples
import plotly.graph_objects as go  # graphiques avancés
from sklearn.linear_model import LinearRegression  # modèle de régression
import statsmodels.api as sm  # stats (ANOVA)
from statsmodels.formula.api import ols  # modèle statistique

# -----------------------------
# CONFIGURATION DE LA PAGE
# -----------------------------
st.set_page_config(page_title="Analyse de plans d'expérience", layout="wide")

# Style visuel (couleurs + police)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1 {color:#0B3C5D; font-weight:700;}
h2 {color:#1D65A6; font-weight:600;}
h3 {color:#2E8BC0; font-weight:500;}
.stButton>button { background-color: #1D65A6; color: white; border-radius: 8px; padding: 0.5em 1em; font-weight: 600;}
.stButton>button:hover { background-color: #145a96; color: #fff; }
.stApp { background: linear-gradient(180deg, #f0f8ff 0%, #ffffff 100%); color: #0B3C5D; }
</style>
""", unsafe_allow_html=True)

# Titre de l'app
st.title("Outil d'analyse DOE")
st.markdown("**Zoé, Anaïs et Bastien** – FIP MECA 4")

# 👉 MESSAGE DE BIENVENUE AJOUTÉ ICI
st.markdown("""
### 👋 Bienvenue sur l'application d'analyse de plans d'expérience (DOE)

Cette application vous permet de :
- 📊 Explorer vos données facilement  
- 📈 Visualiser les effets des facteurs  
- 🔗 Identifier les interactions  
- 📉 Évaluer la qualité d’un modèle  
- 📊 Réaliser une analyse ANOVA  

👉 **Pour commencer :**  
Chargez un fichier de données (CSV ou Excel) dans le menu à gauche, puis sélectionnez vos variables.

💡 Cet outil est conçu pour vous aider à comprendre et interpréter rapidement vos plans d'expérience.
""")

# -----------------------------
# FONCTIONS UTILES
# -----------------------------

def load_file(file):
    # charge un fichier csv ou excel
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier : {e}")
        st.stop()

def plot_histograms(data, numeric_cols, palette):
    # affiche histogrammes + boxplots
    for col in numeric_cols:
        fig = px.histogram(data, x=col, marginal="box", nbins=20,
                           color_discrete_sequence=[palette[2]],
                           title=f"Histogramme et boxplot de {col}")
        st.plotly_chart(fig, use_container_width=True)

def plot_effects(data, facteurs, reponse, palette):
    # effet de chaque facteur sur la réponse
    for i, f in enumerate(facteurs):
        means = data.groupby(f)[reponse].mean().reset_index()
        fig = px.line(means, x=f, y=reponse, markers=True,
                      title=f"Effet principal : {f}",
                      color_discrete_sequence=[palette[i % len(palette)]])
        st.plotly_chart(fig, use_container_width=True)

def plot_interactions(data, facteurs, reponse):
    # interaction entre deux facteurs
    if len(facteurs) >= 2:
        f1 = st.selectbox("Facteur X", facteurs, key="f1_int")
        f2 = st.selectbox("Facteur lignes", [f for f in facteurs if f != f1], key="f2_int")

        fig = go.Figure()
        for level in sorted(data[f2].unique()):
            subset = data[data[f2] == level]
            means = subset.groupby(f1)[reponse].mean().sort_index()
            fig.add_trace(go.Scatter(x=means.index, y=means.values,
                                     mode='lines+markers', name=str(level)))

        fig.update_layout(title=f"Interactions entre {f1} et {f2}",
                          xaxis_title=f1, yaxis_title=reponse)
        st.plotly_chart(fig, use_container_width=True)

def plot_surface3d(X, y, facteurs, reponse):
    # surface 3D (2 facteurs)
    if len(facteurs) >= 2:
        f1, f2 = facteurs[:2]

        x1 = np.linspace(X[f1].min(), X[f1].max(), 30)
        x2 = np.linspace(X[f2].min(), X[f2].max(), 30)
        X1, X2 = np.meshgrid(x1, x2)

        model2 = LinearRegression().fit(X[[f1, f2]], y)
        Z = model2.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

        fig = go.Figure(data=[go.Surface(z=Z, x=X1, y=X2, colorscale='Blues', opacity=0.8)])
        fig.add_trace(go.Scatter3d(x=X[f1], y=X[f2], z=y,
                                   mode='markers', marker=dict(color='darkblue', size=4)))

        fig.update_layout(scene=dict(xaxis_title=f1, yaxis_title=f2, zaxis_title=reponse))
        st.plotly_chart(fig, use_container_width=True)

def display_anova(data, facteurs, reponse):
    # calcul ANOVA
    formula = reponse + " ~ " + " + ".join(facteurs)
    lm = ols(formula, data=data).fit()
    anova = sm.stats.anova_lm(lm, typ=2)

    anova['Significatif'] = np.where(anova['PR(>F)'] < 0.05, "Oui", "Non")
    anova = anova.sort_values(by='PR(>F)')

    fill_colors = []
    for col in anova.columns:
        col_colors = []
        for i in range(len(anova)):
            if anova['Significatif'].iloc[i] == "Oui":
                col_colors.append('#d4f4dd')
            else:
                col_colors.append('#f0f0f0')
        fill_colors.append(col_colors)

    fig = go.Figure(data=[go.Table(
        header=dict(values=list(anova.columns), fill_color='#1D65A6',
                    font=dict(color='white', size=14), align='left'),
        cells=dict(values=[anova[col] for col in anova.columns],
                   fill_color=fill_colors,
                   align='left',
                   font=dict(color='black', size=12))
    )])
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# COULEURS
# -----------------------------
palette = ['#0B3C5D','#1D65A6','#2E8BC0','#3EA0C9','#6BB1D1','#91C0E0','#B0D4EB','#C9E2F3','#E1F0F8','#F0FAFF']

# -----------------------------
# CHARGEMENT DES DONNÉES
# -----------------------------
file = st.sidebar.file_uploader("📂 Charger un fichier de données (CSV ou Excel)")

if file:
    df = load_file(file)

    st.subheader("Aperçu des données")
    st.dataframe(df)

    cols = df.columns.tolist()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.sidebar.header("Sélection des variables")
    facteurs = st.sidebar.multiselect("Facteurs (variables d'entrée)", cols, default=num_cols[:-1])
    reponse = st.sidebar.selectbox("Réponse (variable à expliquer)", cols)

    # ⚠️ message si erreur
    if reponse in facteurs:
        st.warning(f"""
⚠️ **Attention : incohérence dans les variables sélectionnées**

La variable **{reponse}** est utilisée à la fois comme facteur et comme réponse.

👉 Le modèle va expliquer une variable par elle-même → résultats peu fiables.

💡 Retire cette variable des facteurs.
""")

    st.sidebar.header("Graphiques supplémentaires")
    show_hist = st.sidebar.checkbox("Afficher histogrammes / boxplots", value=True)
    show_corr = st.sidebar.checkbox("Afficher matrice de corrélation", value=True)

    if facteurs and reponse:
        data = df[facteurs + [reponse]].dropna()
        X = data[facteurs]
        y = data[reponse]

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Résumé données", "Qualité modèle", "Effets", "Interactions", "3D", "ANOVA"
        ])

        with tab1:
            st.write(data.describe())
            if show_corr:
                st.dataframe(data.corr())
            if show_hist:
                num_cols_data = [col for col in num_cols if col in data.columns]
                plot_histograms(data, num_cols_data, palette)

        with tab2:
            try:
                model = LinearRegression().fit(X, y)
                yp = model.predict(X)
            except Exception as e:
                st.error(f"Erreur modèle : {e}")
                st.stop()

            equation = f"{reponse} = {model.intercept_:.3f} " + " ".join(
                [f"+ ({c:.3f}*{f})" for c, f in zip(model.coef_, facteurs)]
            )
            st.markdown(f"**Équation :** {equation}")
            st.markdown(f"**R² = {model.score(X, y):.3f}**")

            fig = px.scatter(x=y, y=yp)
            st.plotly_chart(fig, use_container_width=True)

        with tab3:
            plot_effects(data, facteurs, reponse, palette)

        with tab4:
            plot_interactions(data, facteurs, reponse)

        with tab5:
            plot_surface3d(X, y, facteurs, reponse)

        with tab6:
            display_anova(data, facteurs, reponse)

else:
    st.info("Charge un fichier pour commencer")

st.caption("Outil DOE")
