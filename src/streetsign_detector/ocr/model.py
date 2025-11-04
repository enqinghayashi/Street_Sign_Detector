import torch.nn as nn
import torchvision.models as models

from streetsign_detector.constants import MAX_LEN, NUM_CLASS

class OCRModel(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Load Pretrained EfficientNet
        try:
            # Newer Torchvision versions
            weights = models.EfficientNet_B0_Weights.DEFAULT
            base = models.efficientnet_b0(weights=weights)
        except:
            # Older Torchvision versions fallback
            base = models.efficientnet_b0(pretrained=True)

        # EfficientNet B0 final feature map size is 1280
        self.backbone = base



