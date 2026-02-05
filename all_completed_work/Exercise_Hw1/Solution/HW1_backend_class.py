import torch
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image

# Class to specifically convert grayscale to RGB for VGG model
class GrayscaleToRGB(object):
    def __call__(self, img):
        if img.mode == 'L':  # Grayscale
            img = img.convert('RGB')
        return img

# Class to load models, preprocess images for transforms and predict
class MyPredictor: 
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self,
        image = None,
        model_path = None,
        model_type = 'vgg',
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        """
        Predicts the class of a CT scan image using a saved model.

        Args:
            image_path (str): Path to the input image.
            model_path (str): Path to the saved model weights.
            model_type (str): 'vgg' or 'resnet' (default: 'resnet').
            input_channels (str): 'grayscale' or 'rgb' (default: 'grayscale').
            device (str): 'cuda' or 'cpu' (default: 'cuda').

        Returns:
            str: Predicted class ('COVID' or 'Normal').
            float: Confidence score (0-1).
        """
        # Load the model as per user's choice and preprocess
        if model_type == 'vgg':
            model_path = 'vgg16_covid_transfer_learning.pth'
            model = models.vgg16(weights=None)
            model.classifier[6] = nn.Linear(4096, 2)
            # if input_channels == 'grayscale':
            #     model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # For grayscale
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            preprocess = transforms.Compose([
                GrayscaleToRGB(),  # Add this first
                transforms.Resize((256, 256)),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
            img_tensor = preprocess(image).unsqueeze(0).to(device)

        elif model_type == 'resnet':
            model_path='resnet50_covid_transfer_learning.pth'
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(2048, 2)
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # For grayscale   
            model.load_state_dict(torch.load(model_path, map_location=device))
            model = model.to(device)
            model.eval()
            
            preprocess = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            img_tensor = preprocess(image).unsqueeze(0).to(device)
        else:
            raise ValueError("model_type must be 'vgg' or 'resnet'")

        # Predict
        with torch.no_grad():
            outputs = model(img_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.nn.functional.softmax(outputs, dim=1)

        class_names = ['COVID', 'Normal']
        return class_names[preds[0]], probs[0][preds[0]].item()