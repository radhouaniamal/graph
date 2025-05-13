import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Étape 1 : Charger les données 
data = pd.read_csv('CA-GrQc.txt', sep='\t', skiprows=4, names=['FromNodeId', 'ToNodeId'])

# Étape 2 : Construire le graphe non dirigé
G = nx.Graph()
G.add_edges_from(data[['FromNodeId', 'ToNodeId']].values)

# Étape 3 : Visualiser le graphe (optionnel)
plt.figure(figsize=(10, 10))
nx.draw(G, with_labels=False, node_size=10, node_color='blue', edge_color='gray', alpha=0.5)
plt.title("Réseau de collaboration CA-GrQc")
plt.show()

# Étape 4 : Fonction pour extraire les caractéristiques
def extract_features(G, node_pairs):
    features = []
    for u, v in node_pairs:
        # Degré des nœuds
        degree_u = G.degree(u)
        degree_v = G.degree(v)
        
        # Voisins communs
        common_neigh = len(list(nx.common_neighbors(G, u, v)))
        
        # Coefficient de Jaccard
        jaccard = list(nx.jaccard_coefficient(G, [(u, v)]))[0][2]
        
        features.append([degree_u, degree_v, common_neigh, jaccard])
    return np.array(features)

# Étape 5 : Préparer les données pour le machine learning
# Arêtes existantes (positives)
existing_edges = list(G.edges())
features_existing = extract_features(G, existing_edges)

# Générer des arêtes négatives (paires de nœuds sans connexion)
non_edges = list(nx.non_edges(G))
np.random.seed(42)
non_edges_sample = np.random.choice(len(non_edges), size=len(existing_edges), replace=False)
non_edges_selected = [non_edges[i] for i in non_edges_sample]

# Extraire les caractéristiques pour les arêtes négatives
features_non_existing = extract_features(G, non_edges_selected)

# Combiner les données
X = np.vstack((features_existing, features_non_existing))
y = np.hstack((np.ones(len(features_existing)), np.zeros(len(features_non_existing))))

# Étape 6 : Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Étape 7 : Entraîner le modèle Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Étape 8 : Prédire et évaluer le modèle
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Métriques d'évaluation
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")