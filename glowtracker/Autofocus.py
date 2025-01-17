import PIL.Image
import torch
from torchvision import transforms
from Autofocus_Model import *
import configparser
import os
import PIL

def get_img_paths(dirs, extension):
    img_paths = []
    for dir in dirs:
        for f in os.listdir(dir):
            if f.endswith(extension): 
                img_paths.append(os.path.join(dir, f))
    return img_paths

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float32),  # Convert to 32-bit floating point
])

def load_model(model_path, model_type):
    model = get_model(model=model_type)
    model_state_dict = torch.load(model_path, map_location=torch.device('cpu')) 
    model.load_state_dict(model_state_dict)
    return model

def get_autofocus_delta(model, img):
    
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    delta = None
    model.eval()
    with torch.no_grad():
        delta = round((model(img)).item() / 1000, 2)
        print(delta)
    return delta

if __name__ == '__main__':
    config_object = configparser.ConfigParser()
    with open('glowtracker.ini', 'r') as f:
        config_object.read_file(f)
        model_path = config_object.get('Autofocus', 'model_path')
        model_type = config_object.get('Autofocus', 'model_type')
        extension = config_object.get('Experiment', 'extension')
        model = load_model(model_path=model_path, model_type=model_type)
        test_dir = r'C:\Users\rane\Desktop\Thesis\GlowTracker\glowtracker\snaps\Analysis\Test'
        test_paths = get_img_paths([test_dir], extension)
        for path in test_paths:
            image = PIL.Image.open(path).convert('RGB')
            print(f'{path} : [{get_autofocus_delta(model, image)}]')
