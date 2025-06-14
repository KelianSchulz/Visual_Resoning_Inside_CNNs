import os

# Zielbasisverzeichnis
target_base = "heatmap_concepts"

# Mapping von Filter-IDs zu Konzeptnamen (aus deiner Tabelle)
filter_label_map = {
    0: "filter_0_objektsilhouette",
    1: "filter_1_objektvorderseite",
    2: "filter_2_flugzeug_koerperform",
    4: "filter_4_tiermasse",
    6: "filter_6_flaeche_silhouette",
    7: "filter_7_tiergesicht_frontal",
    9: "filter_9_himmel_kontrast",
    11: "filter_11_tierkontur_frontal",
    14: "filter_14_flugzeug_kontur",
    15: "filter_15_flugzeug_koerper_horizontal"
}

# Zielordner erstellen
created_dirs = []
for folder in filter_label_map.values():
    path = os.path.join(target_base, folder)
    os.makedirs(path, exist_ok=True)
    created_dirs.append(path)



