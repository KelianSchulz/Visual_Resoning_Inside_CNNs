import os


classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


base_dir = "original_pics"


for class_name in classes:
    path = os.path.join(base_dir, class_name)
    os.makedirs(path, exist_ok=True)
