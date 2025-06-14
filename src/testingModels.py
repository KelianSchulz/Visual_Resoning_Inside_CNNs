import torch as t
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
from analyze_purity_scores import *
from secondmodel64 import ConceptCNN
from main64 import net, test_set, device, visual_hook, feature_maps  , classes

MODEL_PATH = "./models/cifar_net_64.pth"                  
net.load_state_dict(t.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()
relevant_filters_conv2 = [22, 26, 66, 77, 89, 105]

label_mapping = {
    0: "Animal head",
    1: "Background surface",
    2: "Horizontal edges",
    3: "Local details",
    4: "Surface texture",
    5: "Object body",
    6: "Object silhouette",
    7: "Silhouette contrast",
    8: "Animal body",
    9: "Vehicle structure"
}
concept_inference_transform = transforms.Compose([
    transforms.Resize((64, 64)),       
    transforms.ToTensor(),            
    transforms.Normalize((0.5,), (0.5,))  
])

concept_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

def load_concept_model(path="./models/best_concept_net.pth"):
    model = ConceptCNN().to(device)
    model.load_state_dict(t.load(path))
    model.to(device)
    model.eval()
    return model

def extract_selected_heatmaps(image: t.Tensor, selected_filters: list[int]) -> list[t.Tensor]:
    feature_maps.clear()  # vorher leeren
    net.conv2.register_forward_hook(visual_hook)

    image_batch = image.to(device)  # [1, 3, 64, 64]
    with t.no_grad():
        _ = net(image_batch)

    fmap = feature_maps[0]  # fmap shape: [1, 128, H, W] z. B. [1, 128, 13, 13]
    
    selected_maps = []
    for i in selected_filters:
        fmap_i = fmap[0, i, :, :]  # [H, W] z. B. [13, 13]
        fmap_resized = t.nn.functional.interpolate(
            fmap_i.unsqueeze(0).unsqueeze(0),  # → [1, 1, H, W]
            size=(64, 64),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # → [1, 64, 64]
        selected_maps.append(fmap_resized)

    return selected_maps 



def classify_concepts(heatmaps, concept_model):
    predictions, confs = [], []

    for heatmap in heatmaps:
        pil = to_pil_image(heatmap.squeeze(0), mode= "F")
        input_tensor = concept_inference_transform(pil).unsqueeze(0).to(device)  # -> [1, 3, 64, 64]
        
        with t.no_grad():
            output = concept_model(input_tensor)
            probability = t.softmax(output, dim=1)[0]
            predicted_label = int(probability.argmax())
            predictions.append(predicted_label)
            confs.append(probability[predicted_label].item())

    return predictions, confs

def classify_image_with_main_model(image_tensor):
    image_tensor = image_tensor.to(device)
    
    net.eval()
    with t.no_grad():
        output = net(image_tensor)
        predicted_index = output.argmax(dim=1).item()
    
   
    return classes[predicted_index]



with open("purity_scores.json", "r", encoding="utf-8") as f:
    purity_scores = json.load(f)




def explain_image(image_tensor, selected_filters, concept_model):
    predicted_class = classify_image_with_main_model(image_tensor)
    heatmaps = extract_selected_heatmaps(image_tensor, selected_filters)
    predictions, confs = classify_concepts(heatmaps, concept_model)
    concept_labels = [label_mapping[i] for i in predictions]

 
    counts = Counter(concept_labels)
    top_labels = [lbl for lbl, _ in counts.most_common()]

  
    avg_probs = {}

    for lbl in top_labels:
        ps = []
        for i in range(len(concept_labels)):
            if concept_labels[i] == lbl:
                ps.append(confs[i])
        
        if ps:
            avg_probs[lbl] = sum(ps) / len(ps)
        else:
            avg_probs[lbl] = 0.0

   
    explanation = []
    for lbl in top_labels:
        score = avg_probs[lbl] * 100
        if lbl in expected.get(predicted_class, []):
            status = f"erwartet für Klasse {predicted_class}"
        elif purity_scores.get(lbl, {}).get(predicted_class, 0) < 0.15:
            status = f"untypisch, passt nicht zu {predicted_class}"
        else:
            status = "neutral"
        explanation.append(f"{lbl} ({score:.1f}%) {status}")

    print(f"The Picture is '{predicted_class}' .")
    print(f"Explanation:")
    for line in explanation:
        print("  ", line)

    return predicted_class, concept_labels, heatmaps, predictions, confs, explanation


def compare_show_plot():
    original = image_tensor.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    original = (original * 0.5) + 0.5  # [-1,1] → [0,1]

    n = len(explanation)
    plt.figure(figsize=(2.5 * (n + 1), 4))  # Dynamische Breite

    # Originalbild
    plt.subplot(1, n + 1, 1)
    plt.imshow(original)
    plt.axis("off")
    plt.title(f"Original ({true_class})")

    # Dynamische Heatmaps der Top-Konzepte
    for i in range(n):
        heatmap_img = heatmaps[i].squeeze().detach().cpu().numpy()
        plt.subplot(1, n + 1, i + 2)
        plt.imshow(heatmap_img)
        plt.axis("off")
        plt.title(explanation[i])

    title = f"Prediction: {predicted_class}\n\n📌 Explanation:\n" + "\n".join(explanation)
    
    plt.suptitle(title, fontsize=10, y=0.98)  
    plt.tight_layout(rect=(0, 0, 1, 0.90))  
    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    idx = np.random.randint(0, len(test_set))
    image_tensor = test_set[idx][0].to(device).unsqueeze(0)
    true_index = test_set[idx][1]
    true_class = classes[true_index]
    
   
    concept_model = load_concept_model()

    predicted_class, concept_labels, heatmaps, predictions, confs, explanation = explain_image(
    image_tensor, relevant_filters_conv2, concept_model)


    compare_show_plot()
