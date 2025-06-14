import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json


#------------------Erwartungsmatrix Definieren ------------------


expected = {

    "plane": ["Object silhouette", "Horizontal edges", "Surface texture", "Background surface", "Silhouette contrast", "Object body", "Local details"],
    "car": ["Object silhouette", "Horizontal edges", "Surface texture", "Local details", "Object body"],
    "bird": ["Animal body", "Object silhouette", "Surface texture"],
    "cat": ["Animal body", "Animal head", "Object silhouette", "Surface texture", "Local details"],
    "deer": ["Animal body", "Animal head", "Object silhouette"],
    "dog": ["Object silhouette", "Animal body", "Animal head", "Surface texture"],
    "frog": ["Surface texture", "Object silhouette", "Animal body"],
    "horse": ["Surface texture", "Object body", "Object silhouette", "Animal head", "Animal body"],
    "ship": ["Vehicle structure", "Background surface", "Horizontal edges", "Surface texture", "Object body", "Object silhouette", "Silhouette contrast"],
    "truck": ["Vehicle structure", "Horizontal edges", "Local details", "Object body", "Surface texture", "Object silhouette", "Silhouette contrast"]



}




# ------------------Load CSV ------------------

df = pd.read_csv("./data/concept_labels.csv")  #Path
df["class_name"] = df["filepath"].apply(lambda x: os.path.basename(x).split("_")[0]) #get class name car_0_f77.png --> car


# ------------------Konzept-Klassen-Zählung ------------------


counts = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    concept = row["concept_label"]
    klass = row["class_name"]
    counts[concept][klass] += 1

# ------------------Relative Scores berechnen ------------------

purity_scores = defaultdict(dict)

for concept, class_counts in counts.items():
    total = sum(class_counts.values())
    for klass, count in class_counts.items():
        purity_scores[concept][klass] = round(count / total, 3)

# In DataFrame umwandeln
df_purity = pd.DataFrame(purity_scores).T.fillna(0)
df_purity = df_purity.sort_index(axis=1)  # Klassen alphabetisch sortieren


# ------------------ Optional – Heatmap plotten ------------------


with open("./results/purity_scores.json", "w", encoding="utf-8") as f:
    json.dump(purity_scores, f, indent=2, ensure_ascii=False)
