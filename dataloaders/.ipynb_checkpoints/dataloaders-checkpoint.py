from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import os
import random

class Food101Dataset(Dataset):
    def __init__(self, json_path, root_dir, transform=None):
        """
        Args:
            json_path (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        with open(json_path, 'r') as file:
            self.data = json.load(file)
        
        # Read class names from a .txt file
        with open("./data/food-101/meta/classes.txt", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}

        # Convert the JSON structure to a list of image paths and corresponding labels
        self.image_paths = []
        self.labels = []
        for label, filenames in self.data.items():
            for filename in filenames:
                self.image_paths.append(filename)  # Assuming the filename already has the structure class_name/filename.jpg
                self.labels.append(self.class_to_idx[label])

        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_paths[idx] + ".jpg")
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_random_samples(self, num_samples=5):
        random_indices = random.sample(range(len(self.image_paths)), num_samples)
        images = []
        labels = []
        
        for idx in random_indices:
            image, label = self.__getitem__(idx)
            images.append(image)
            labels.append(label)
            
        return images, labels
