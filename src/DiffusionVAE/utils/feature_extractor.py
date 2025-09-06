import torch
import torchvision.models as models

""" Pre-trained feature extraction Vgg16 model"""
class VggFeatureExtractor:
    def __init__(self, device):
        self.device = device
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = torch.nn.Sequential(*list(vgg.features.children())[:16])  # Up to conv3_3
        self.features.eval()
        self.features.to(device)
        for param in self.features.parameters():
            param.requires_grad = False

    def extract_features(self, x):
        with torch.no_grad():
            features = self.features(x)
        return features