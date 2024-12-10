import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms 


class ImagePromptDataset(Dataset):
    def __init__(self, dataset, transform=None, device = torch.device("cuda"), maps = None):
        """
        Initialize the dataset with optional map types (e.g., depth, edge).
        
        Args:
            dataset: Original dataset containing images and text prompts.
            transform: Optional transformations to apply to images.
            device: Device to load images and maps onto.
            maps: List of map types to generate, e.g., ["depth", "edge"].
                currently only supports 'depth' and 'edge'
        """
        self.dataset = dataset
        self.transform = transform
        self.device = device
        self.maps = maps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        image = sample['image']
        prompt = sample.get('text')## for testing
        
        mappings = self.get_maps(image) if self.maps is not None else {}
        
        if self.transform:
            image = self.transform(image).to(self.device)
            
            if 'depth' in mappings:
                mappings['depth'] = self.transform(mappings['depth']).to(self.device)
            if 'edge' in mappings:
                mappings['edge'] = self.transform(mappings['edge']).to(self.device)
        
        if self.maps is not None:
            return image, prompt, torch.stack([i for i in list(mappings.values())]).squeeze(1)
 
        return image, prompt
    
    def get_maps(self, image):
        """
        Generate specified maps for the image.
        
        Args:
            image: Input image for which maps are generated.

        Returns:
            A dictionary containing requested maps.
        """
        mappings = {}
        
        if 'depth' in self.maps:
            mappings['depth'] = self.calc_depth(image)
        if 'edge' in self.maps:
            mappings['edge'] = self.calc_edge(image)
        
        return mappings
    
    def calc_depth(self, image):
        """
        Calculate depth map for the image. For now we'll utilize MiDaS,
        but others should be considered
        
        Args:
            image: Input image for which depth map is generated.

        Returns:
            Depth map for the image.
        """
        model_type = "DPT_Large"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.eval()
                
        midas.to(self.device)
        transform = torch.hub.load("intel-isl/MiDaS", "transforms").dpt_transform
        
        input_batch = transform(np.array(image.convert("RGB"))).to(self.device)
        with torch.no_grad():
            depth_map = midas(input_batch)
        
        depth_map = (depth_map - torch.min(depth_map))/(torch.max(depth_map) - torch.min(depth_map))
        
        return transforms.ToPILImage()(depth_map)
    
    def calc_edge(self, image, low_threshold = 100, high_threshold = 200):
        """
        Calculate edge map for the image. For now we'll utilize Canny,
        but others should be considered. Testing around thresholds must 
        be done to determine optimal values.
        
        Args:
            image: Input image for which edge map is generated.
            low_threshold: Lower bound for the Canny edge detector hysteresis procedure.
            high_threshold: Upper bound for the Canny edge detector hysteresis procedure.


        Returns:
            Edge map for the image.
        """
        image = np.array(image.convert("RGB"))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        edge_map = cv2.Canny(image, 100, 200)
        
        edge_map = Image.fromarray(edge_map)
        
        return edge_map
        