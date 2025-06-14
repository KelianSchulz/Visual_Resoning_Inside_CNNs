import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision as tv
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
device = t.device("cuda" if t.cuda.is_available() else "cpu")

batch_size = 64 

transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(64, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
 
    ])


train_set = tv.datasets.CIFAR10(root="./data", train=True, download=False, transform=transform)
trainloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)

test_set = tv.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
testloader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=2)


classes = ('plane', 'car', 'bird', 'cat',
                'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
activations = {}



#-------------------------Collector/Hook-----------------------------------------
class ActivationCollector:
        def __init__(self): 
            self.activations = [] 
            self.current_label = None

        def hook (self, modul, input, output):
            if self.current_label is None:
                return
            for i in range (output.shape[0]): # for each batch
                for n in range(output.shape[1]): # for each filter 
                    wert = output[i,n].mean().item() # get avg activation
                    label = self.current_label[i].item() # get label
                    self.activations.append((label, n, wert)) # append label, filter, activation      

def visual_hook(module, input, output):
    feature_maps.append(output.detach().cpu())

#-------------------------Model-----------------------------------------
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels= 3, out_channels= 16, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels= 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 13 * 13, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
       
    def forward(self, model):
        model = self.pool(F.relu(self.bn1(self.conv1(model))))
        model = self.pool(F.relu(self.bn2(self.conv2(model))))
        model = t.flatten(model, 1)    
        model = F.relu(self.fc1(model))
        model = F.relu(self.fc2(model))
        model = self.fc3(model)
        return model    

net = Net().to(device)
plane_images = []
feature_maps = []


#-------------------------Train-----------------------------------------


def train():
    criterion = nn.CrossEntropyLoss()
    optimizer = t.optim.Adam(net.parameters(), lr=0.0003)

    for epoch in range(50):
        running_loss = 0.0
        batch_counter = 0  

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data 
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            batch_counter += 1

            if batch_counter == 10:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
                batch_counter = 0  

            PATH = './models/cifar_net_64.pth'
            t.save(net.state_dict(), PATH)


#-------------------------Test-----------------------------------------


def test():
    net.load_state_dict(t.load('./models/cifar_net_64.pth'))
    net.to(device)
    net.eval()
    print("Netzwerk wurde geladen und in Eval-Modus gesetzt.")
    correct = 0
    total = 0
    batch_count = 0

    for data in testloader:
        batch_count += 1
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

       
                

        collector.current_label = labels
        outputs = net(inputs) # torch.Size([64, 10])
        predicted = outputs.argmax(dim=1) # Tensor mit dem Index der Predicteten klasse: Bei 64: torch.Size([0,2,3,5,8,...x63])
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    
    accuracy = 100 * correct / total
    print(f"Genauigkeit auf dem Testset: {accuracy:.2f}%")
    print(f"Anzahl der Test-Batches: {batch_count}")

#-------------------------Visualizing-----------------------------------------

def visualize_by_label(label_name, filter_index, n_images):
    feature_maps.clear()

    label_index = classes.index(label_name)
    found_images = []

    for data in testloader:
        inputs, labels = data
        for i in range(inputs.shape[0]):
            if labels[i].item() == label_index:
                found_images.append(inputs[i])
                if len(found_images) >= n_images:
                    break
        if len(found_images) >= n_images:
                break        
                
    # 2. Visualisieren
    for idx, image in enumerate(found_images):
        net.conv2.register_forward_hook(visual_hook)
        with t.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            _ = net(image_batch)

        fmap = feature_maps[0][0, filter_index].cpu().numpy()
        fmap_resized = cv2.resize(fmap, (32, 32))

        orig = image.squeeze().permute(1, 2, 0).cpu().numpy()
        orig = (orig + 1) / 2

        plt.figure(figsize=(8, 4))

        # Linkes Bild: Original
        plt.subplot(1, 2, 1)
        plt.imshow(orig)
        plt.title("Originalbild")
        plt.axis("off")

        # Rechtes Bild: Heatmap
        plt.subplot(1, 2, 2)
        plt.imshow(fmap_resized, cmap="hot")
        plt.colorbar()
        plt.title(f"Filter {filter_index} – Heatmap")
        plt.axis("off")

        plt.suptitle(f"{label_name.capitalize()} – Bild {idx+1}")
        plt.tight_layout()
        plt.show()


        feature_maps.clear()  # wichtig fürs nächste Bild            


#-------------------------Extract Heatmaps-----------------------------------------

def extract_all_heatmaps(n_per_class=100):
    import os
    heatmap_data = []  

    for class_name in classes:
        os.makedirs(f"heatmapsConv2/{class_name}", exist_ok=True)
        os.makedirs(f"original_pics/{class_name}", exist_ok=True)
   
    for label_index in range(10):
        class_name = classes[label_index]
        found_images = []
        saved = 0

   
        for data in testloader:
            inputs, labels = data
            for i in range(inputs.shape[0]):
                if labels[i].item() == label_index:
                    found_images.append(inputs[i])
                    saved += 1
                    if saved >= n_per_class:
                        break
            if saved >= n_per_class:
                break

        for idx, img in enumerate(found_images):
            feature_maps.clear()  
            net.conv2.register_forward_hook(visual_hook)
            with t.no_grad():
                _ = net(img.unsqueeze(0).to(device))
         

            orig = img.cpu().permute(1, 2, 0).numpy() 
            orig = (orig + 1) / 2  

            orig_uint8 = (orig * 255).astype(np.uint8)  

            original_path = f"original_pics/{class_name}/{class_name}_{idx}.png"
            cv2.imwrite(original_path, cv2.cvtColor(orig_uint8, cv2.COLOR_RGB2BGR))

            for filter_index in range(128):
               
                fmap = feature_maps[0][0, filter_index].numpy()
                fmap_resized = cv2.resize(fmap, (64, 64))
                fmap_norm = (fmap_resized - fmap_resized.min()) / (fmap_resized.ptp() + 1e-5)
                fmap_uint8 = (fmap_norm * 255).astype(np.uint8)


                filename = f"{class_name}_{idx}_f{filter_index}.png"
                path = f"./data/heatmapsConv2/{class_name}/{filename}"
                colored_fmap = cv2.applyColorMap(fmap_uint8, cv2.COLORMAP_JET)
                cv2.imwrite(path, colored_fmap)

                heatmap_data.append([filename, class_name, label_index, filter_index])

    df = pd.DataFrame(heatmap_data, columns=["filename", "class", "class_index", "filter"])
    df.to_csv("heatmap_labelsConv2.csv", index=False)
    print("Alle Heatmaps extrahiert und gespeichert.")

def evaluate_single_image_single_filter(label_index, filter_index=None, image_index=0):
    found_images = []

    for data in testloader:
        inputs, labels = data
        for i in range(inputs.shape[0]):
            if labels[i].item() == label_index:
                found_images.append(inputs[i])
            if len(found_images) > image_index:
                break
        if len(found_images) > image_index:
            break

    image = found_images[image_index].to(device)
    true_label = label_index

    def filter_mask_hook(module, input, output):
        output_clone = output.clone()
        if filter_index is not None:          
            for i in range(output_clone.shape[1]):
                if i != filter_index:
                    output_clone[:, i, :, :] = 0
        return output_clone

    hook_handle = net.conv2.register_forward_hook(filter_mask_hook)

    net.eval()
    with t.no_grad():
        output = net(image.unsqueeze(0))
        predicted_label = output.argmax(dim=1).item()

    hook_handle.remove()

    print(f" Wahre Klasse:     {classes[true_label]}")
    print(f" Vorhersage mit **nur Filter {filter_index}**: {classes[predicted_label]}")

    img_np = image.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np + 1) / 2
    plt.imshow(img_np)
    plt.title(f"Wahr: {classes[true_label]} – Vorhergesagt: {classes[predicted_label]}")
    plt.axis("off")
    plt.show()

def extract_heatmaps_from_image(image_tensor, model, selected_layer, target_size=(64, 64)):
    feature_maps = []

    def hook(module, input, output):
        feature_maps.append(output)

    handle = selected_layer.register_forward_hook(hook)

    with t.no_grad():
        model.eval()
        _ = model(image_tensor.unsqueeze(0).to(device))

    handle.remove()

    maps = feature_maps[0].squeeze(0).cpu()
    heatmaps = []
    for fmap in maps:
        fmap_resized = F.interpolate(fmap.unsqueeze(0).unsqueeze(0), size=target_size, mode="bilinear", align_corners=False)
        heatmaps.append(fmap_resized.squeeze(0))

    return heatmaps



concept_images = [4, 8, 16, 17, 29, 31, 33, 35, 39, 40, 42, 46, 49, 57, 58, 64, 77, 78, 84, 91]
non_concept_images = [1, 2, 3, 6, 9, 10, 11, 14, 15, 18, 21, 22, 24, 27, 28, 34, 37, 44, 45, 47]




def get_conv2_vector(image_tensor):
    feature_maps.clear()
    net.eval()
    with t.no_grad():
        _= net(image_tensor.unsqueeze(0).to(device))
    fmap = feature_maps[0]
    return fmap.view(-1).numpy() # Tensor to Vector

def zeige_bild(idx):
    bild, label = test_set[idx]
    np_img = bild.permute(1, 2, 0).numpy()
    np_img = (np_img + 1) / 2  # falls du (0.5, 0.5, 0.5) normalisiert hast

    plt.imshow(np_img)
    plt.title(f"Index: {idx} – Klasse: {classes[label]}")
    plt.axis("off")
    plt.show()

def zeige_alle_bilder_aus_klasse(label_name, n=20):
    target_index = classes.index(label_name)
    count = 0
    for idx in range(len(test_set)):
        _, label = test_set[idx]
        if label == target_index:
            zeige_bild(idx)
            print(f"Index = {idx}")
            count += 1
        if count >= n:
            break

if __name__ == '__main__':
    
    collector = ActivationCollector()
    #net.conv2.register_forward_hook(collector.hook)
    #train()
    #test() 
    #print("Gespeicherte Einträge:", len(collector.activations))
    net.conv2.register_forward_hook(visual_hook)  
    

    X = []
    y = []

    for idx in tqdm(concept_images, desc="With Concept"):
        img, _ = test_set[idx]
        X.append(get_conv2_vector(img))
        y.append(1)

    for idx in tqdm(non_concept_images, desc="Ohne Konzept"):
        img, _ = test_set[idx]
        X.append(get_conv2_vector(img))
        y.append(0)
   
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)

    cav_vector = clf.coef_[0]
    # test_idx = 741  # irgendein Bild aus test_set
    # image, _ = test_set[test_idx]    
    # vec = get_conv2_vector(image)  # dein flatten(conv2)
    # score = np.dot(vec, cav_vector)
    # zeige_alle_bilder_aus_klasse("plane", n=5)
    class_images = {
    "dog":     [12, 16, 24, 31, 33],
    "cat":     [0, 8, 46, 53, 61],
    "deer":    [22, 26, 32, 36, 40],
    "truck":   [11, 14, 23, 28, 34],
    "ship":    [1, 2, 15, 18, 51],
    "plane":   [3, 10, 21, 27, 44]
    }

    class_to_scores = {}
    for klasse, indices in class_images.items():
        scores = []

        for idx in indices:
            img, _ = test_set[idx]
            vec = get_conv2_vector(img)           
            score = np.dot(vec, cav_vector)       
            scores.append(score)

        class_to_scores[klasse] = scores

    mean_scores = {klasse: np.mean(scores) for klasse, scores in class_to_scores.items()}




    plt.figure(figsize=(10, 5))
    plt.bar(list(mean_scores.keys()), list(mean_scores.values()))
    plt.title("Average Animal-Head-CAV-Score per Class")
    plt.xlabel("Klasse")
    plt.ylabel("CAV-Score ('Animal-Head')")
    plt.axhline(0, color='gray', linestyle='--')
    plt.grid(axis="y", linestyle=":", linewidth=0.7)
    plt.show()        