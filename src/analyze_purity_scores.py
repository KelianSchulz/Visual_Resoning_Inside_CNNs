import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import json


#------------------Erwartungsmatrix Definieren ------------------

expected = {

"plane": ["Objektsilhouette", "Horizontale Kanten", "Oberflächentextur", "Hintergrundfläche", "Silhouettenkontrast", "Objekt Körper", "Lokale Details"],
"car": ["Objektsilhouette", "Horizontale Kanten", "Oberflächentextur", "Lokale Details", "Objektkörper"],
"bird" : ["Tierkörper", "Objektsilhouette", "Oberflächentextur"],
"cat" : ["Tierkörper", "Tierkopf", "Objektsilhouette", "Oberflächentextur", "Lokale Details"],
"deer": ["Tierkörper", "Tierkopf", "Objektsilhouette"],
"dog" : ["Objektsilhouette", "Tierkörper", "Tierkopf", "Oberflächentextur"],
"frog" : ["Oberflächentextur", "Objektsilhouette", "Tierkörper"],
"horse" : ["Oberflächentextur", "Objektkörper", "Objektsilhouette", "Tierkopf", "Tierkörper"],
"ship" : ["Fahrzeugstruktur", "Hintergrundfläche", "Horizontale Kanten", "Oberflächentextur", "Objektkörper", "Objektsilhouette", "Silhouettenkontrast"], 
"truck": ["Fahrzeugstruktur", "Horizontale Kanten", "Lokale Details", "Objektkörper", "Oberflächentextur", "Objektsilhouette", "Silhouettenkontrast"]




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
