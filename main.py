import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier
import lime
import lime.lime_tabular
import pickle
import joblib
from joblib import dump, load


import streamlit as st

# --------------------------------------------------------------
# -------- Importation des données et modèles ------------------
# --------------------------------------------------------------

# Charger les données
X_train = pd.read_csv("X_train.csv", sep=',').head(20)
X_train = X_train.drop('Unnamed: 0', axis=1)

y_train = pd.read_csv("y_train.csv", sep=',').head(20)
y_train = y_train.drop('Unnamed: 0', axis=1)

X_test = pd.read_csv("X_test.csv", sep=',').head(20)
X_test = X_test.drop('Unnamed: 0', axis=1)

# Faire sortir les variables les plus importantes
#model = XGBClassifier()

#model = pickle.load(open("model.pkl", "rb"))

#model = xgb.Booster()
#model.load_model("model.bin")

model = joblib.load('model.joblib')
# fit the model
#model.fit(X_train, y_train)

# --------------------------------------------------------------
# -------------------- Début API -------------------------------
# --------------------------------------------------------------

st.sidebar.write('''
# Objectif de l'application:
Prédire le remboursement du prêt pour un client
''')

st.sidebar.header("Paramètres d'entrée")

# bouton selection données global/client
# Initialise une variable pour stocker la sélection de l'utilisateur
info_selection = "Global"
# Ajoutez un bouton radio pour sélectionner les informations globales ou personnelles
info_selection = st.sidebar.radio("Select Information", ("Global", "Client"))

def user_input():
    id_client = st.sidebar.number_input('ID Client',299977)
    return(id_client)

client = user_input()

def user_data():
    data_client = X_test[X_test["SK_ID_CURR"] == client]
    return (data_client)

df_client = user_data()

# ----------------------- PAGE GLOBAL --------------------------

if info_selection == "Global":
    st.header("Information Global")

# --------------------------------------------------------------

    st.subheader('Global Data')
    st.write(X_test)
    #st.write(X_train)

# --------------------------------------------------------------

    # Obtenez les importances des features
    importances = model.feature_importances_

    # Tri des importances dans l'ordre décroissant
    indices = np.argsort(importances)[::-1]

    # disable a warning message
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Affiche les 20 premières importances sous forme de diagramme à barres horizonal
    st.subheader("Les 20 caractéristiques les plus importantes sont :")
    plt.figure(figsize=(10,5))
    plt.barh(X_train.columns[indices[:20]], importances[indices[:20]], align='center')
    plt.xlabel('Importances')
    plt.ylabel('Noms des features')
    plt.title('Importances des 20 premières features')
    st.pyplot()

# ----------------------- PAGE Client --------------------------

else:
    st.header("Information Client")

# --------------------------------------------------------------

    st.subheader('Numéro client : ')
    st.write(client)

    st.write("Données client")
    st.write(df_client)

    st.subheader("Faut-il accepté le crédit pour ce client ?")
    st.write("0 = Credit Accepté - 1 = Credit Refusé")

    proba = model.predict_proba(df_client)[0][1]
    prediction = round(proba)
    st.write("Prediction:", prediction)
    if prediction == 0:
        st.write("On accepte le crédit")
    else:
        st.write("On refuse le crédit")

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.values.tolist(),
                                                  class_names=y_train, verbose=True, mode='regression')

    if prediction == 0:
        st.subheader("Caractéristiques importantes sur l'avis favoriable")
    else:
        st.subheader("Caractéristiques importantes sur l'avis non favoriable")

    def prediction_client():
        exp = explainer.explain_instance(df_client.values[0], model.predict_proba, num_features=6)
        print(exp.as_list())
        return (exp)

    exp_pred = prediction_client()

    html = exp_pred.as_html()
    import streamlit.components.v1 as components
    components.html(html, height=800)

# --------------------------------------------------------------

    st.subheader('graphique bi-variée avec un code couleur selon le score du client ')

    feature_importances = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_})
    feature_importances.sort_values(by='importance', ascending=False, inplace=True)
    top_20_features = feature_importances.head(20)
    nom_feature = top_20_features['feature']
    list_nom_feature = list(nom_feature)

    st.write("Sélection des variables du graphique bi-variée (Appartenant au 20 meilleurs features importances)")
    # Définir les variables sélectionnées par défaut
    default_var_1 = nom_feature.iloc[0]
    default_var_2 = nom_feature.iloc[1]

    # Créer les boutons de liste déroulante pour sélectionner les variables
    def var_1_select():
        var_1 = st.selectbox("Sélectionnez la première variable", list_nom_feature)
        return (var_1)

    def var_2_select():
        var_2 = st.selectbox("Sélectionnez la deuxième variable", list_nom_feature)
        return (var_2)

    var_1 = var_1_select()
    var_2 = var_2_select()

    #var_1 = st.selectbox("Sélectionnez la première variable", nom_feature, index=default_var_1)
    #var_2 = st.selectbox("Sélectionnez la deuxième variable", nom_feature, index=default_var_2)

    scores = []
    for i in range(len(X_test)):
        row = X_test.iloc[i]
        # Convert the row to a DataFrame
        row_df = row.to_frame()
        # Transpose the single-row DataFrame
        row_df_transposed = row_df.transpose()

        # st.write(row_df_transposed)
        probas = model.predict_proba(row_df_transposed)
        scores.append(probas[:, 1])
    # st.write(scores)

    proba = model.predict_proba(df_client)[:,0]
    st.write("probabilité de remboursement du crédit : ",proba[0])

    # Données à afficher
    x = X_test[var_1]
    y = X_test[var_2]

    # Création du graphique
    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=scores, cmap='viridis')
    ax.set_xlabel(var_1)
    ax.set_ylabel(var_2)
    ax.set_title("Nuage de points : " + var_1 + " en fonction de " + var_2)

    # Ajout de la légende associée au code couleur
    legend1 = ax.legend(*scatter.legend_elements())
    ax.add_artist(legend1)
    # ax.legend(title='Score prédiction')

    # Démarquage du point correspondant à l'ID sélectionné
    ax.scatter(df_client[var_1], df_client[var_2], c='red', marker='x')

    plt.show()
    st.pyplot()

# --------------------------------------------------------------
    # Sortir la position du client par ordre de prédiction

    # Prédire les probabilités pour chaque classe pour chaque client
    probas = model.predict_proba(X_test)
    probas = probas[:,0]

    # Trier les données par ordre de probabilité prédite
    list_proba = probas.tolist()
    list_proba.sort(reverse=True)

    proba = model.predict_proba(df_client)[:,0]
    position = list_proba.index(proba)

    # Afficher la place du client sélectionné
    st.subheader("Positionnement dans l'ordre d'acceptation du crédit")
    st.write("le client est à la position :", position+1)
    st.write("il y a donc potentiellement",position,"client(s) plus fiable(s) pour un crédit.")