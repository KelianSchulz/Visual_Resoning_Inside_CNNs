# 🧠 Visional Reasoning Inside a CNN
A deep learning project exploring interpretable concept-based explanations for image classification models using CNNs and manual filter labeling.

This project is part of a long-term research preparation for high-impact, explainable AI work. It includes concept dataset creation, filter analysis, semantic explanation generation, and visualization using heatmaps.

---

## 📁 Project Structure

```
.
├── models/                     # Trained model checkpoints (.pth)
├── data/                       # Concept labels, metadata, helper CSVs
├── heatmap_concepts/          # Manually labeled concept images
├── heatmaps/                  # Heatmaps of individual filters
├── src/                       # Main Python source code
│   ├── main64.py              # Training & evaluation logic
│   ├── secondmodel64.py       # Concept classifier model
│   ├── testingModels.py  # Testing, scoring, analysis
│   └── makecsv.py, makedir.py ...
├── results/                   # Final output: JSONs, explanations, scores
├── .gitignore
├── README.md
```

---

## 🔍 Core Features

- ✅ Train a CNN on CIFAR-10 from scratch
- ✅ Extract and visualize filter heatmaps
- ✅ Label filters manually using semantic concepts
- ✅ Create a concept-heatmap dataset
- ✅ Train a second model to classify heatmaps into concepts
- ✅ Generate human-readable explanations for predictions

---

## 🛠️ Setup

1. Clone this repository  
   ```
   git clone https://github.com/yourusername/bcai-explainability-agent.git
   cd bcai-explainability-agent
   ```

---

## 🚀 How to Run

### Train main CNN:
```bash
python src/main64.py
```

### Generate Heatmaps
```bash
run extract_all_heatmaps method
```
### Train Concept Classifier:
```bash
python src/secondmodel64.py
```

### Evaluate Purity / Explanations:
```bash
python src/analyze_purity_scores.py
```
### Generate Explanations:
```bash
python src/tesingModels.py
```
---

## 📊 Output

- `purity_scores.json`: Semantic alignment of filters to concepts  
- Visual heatmaps showing what each filter "focuses on"  
- Concept predictions per image explaining **what** the network "sees"

---

## 🧪 Example Concepts

| Concept           | Description                            |
|-------------------|----------------------------------------|
| Animal-Head       | Focus on animal faces                  |
| Vehicle structure | Detects structured mechanical shapes   |
| Object silhouette | Responds to full object outlines       |
| Horizontal edges  | Picks up strong horizontal edges       |

---

## 🧠 Why this Project?

This work is part of a long-term research goal:  
> **Making neural networks more interpretable, structured, and concept-aware.**

I aim to contribute to fundamental AI research in Germany, and build tools that go beyond black-box models.

---

## 📬 Contact

Made with ❤️ by **Kelian Schulz**  
Feel free to reach out via GitHub or [LinkedIn](https://www.linkedin.com/in/kelian-schulz-956836335/)



