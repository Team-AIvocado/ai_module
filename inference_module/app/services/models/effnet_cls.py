import torch
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import timm


class EffnetClassifier:
    def __init__(self, weight_path, device="cpu", class_file=None):
        self.device = device

        if class_file:
            with open(class_file, "r", encoding="utf-8") as f:
                self.class_names = [
                    line.strip() for line in f.readlines() if line.strip()
                ]
        else:
            self.class_names = None

        num_classes = len(self.class_names)

        self.model = timm.create_model(
            "efficientnet_b0", pretrained=False, num_classes=num_classes
        )

        state_dict = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.to(device).eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def predict(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            conf, idx = probs.max(dim=1)

        idx = idx.item()
        conf = conf.item()

        if self.class_names:
            return self.class_names[idx], conf
        return str(idx), conf
