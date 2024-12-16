import argparse
import torch
import torchvision
import torchvision .transforms as T
import numpy as np
from PIL import Image

# Surgical tools
tools ={
    0:'Grasper',
    1:'Bipolar',
    2:'Hook',
    3:'Scissors',
    4:'Clipper',
    5:'Irrigator',
    6:'SpecimenBag',
}

def load_img(image_path):
            image = Image.open(image_path)
            transform = T.Compose([
                T.Resize(224),
                T.ToTensor(), 
                T.Normalize(
                    [0.3456, 0.2281, 0.2233], 
                    [0.2528, 0.2135, 0.2104])])
            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0)
            return input_tensor

def main(args):
     
    # Pretrained ResNet50 model
    model = torchvision.models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
    num_features = model.fc.in_features
    num_classes = len(tools)
    model.fc = torch.nn.Linear(num_features, num_classes)

    # Load fine-tuned model
    checkpoint = torch.load(args.checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    
    input_tensor = load_img(args.image)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.sigmoid(output)

        # Predictions
        print(f'Surgical-tool   Probability (%)')
        print(f'-'*30)
        for i, p in enumerate(probabilities.squeeze().tolist()):
            print(f'{tools[i]:>12} {p*100:>10.0f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFZ-Tool for Surgical Tool Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file")
    parser.add_argument("--image", required=True, help="Path to the input image file")
 
    args = parser.parse_args()
    main(args)