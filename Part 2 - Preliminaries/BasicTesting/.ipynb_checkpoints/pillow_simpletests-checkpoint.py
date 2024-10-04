# Tutorial used for testing: https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/

from torchvision import models # Pre-trained models
from torchvision import transforms # Image pre-processing
import torch
from PIL import Image # Import Pillow
from torchvision.models import AlexNet_Weights

 
#print(dir(models)) # Prints a list of available models included with torchvision to use (tutorial suggests AlexNet)

alexnet = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)


# Image pre-processing
transform = transforms.Compose([            #[1]: defining transformed (processed) image
 transforms.Resize(256),                    #[2] Resize image to 256x256
 transforms.CenterCrop(224),                #[3] Crop image to 224x224 about the center
 transforms.ToTensor(),                     #[4] Convert image to PyTorch Tensor data type
 transforms.Normalize(                      #[5-7] Normalize image
 mean=[0.485, 0.456, 0.406],                
 std=[0.229, 0.224, 0.225]                  
 )])

 # Load input image via Pillow
img = Image.open("SimpleTestImages/stopsign.jpg")

# Pre-process image and prepare a batch to be passed through network
img_t = transform(img)
batch_t = torch.unsqueeze(img_t,0)

# Put model in evaluation mode
alexnet.eval()

# Carry out the inference
out = alexnet(batch_t)

# Read and store the labels from a text file containing 1000 labels
with open('TextFileStorage/imagenet_classes.txt') as f:
  classes = [line.strip() for line in f.readlines()]


# Find label with max score (best prediction)
_, index = torch.max(out, 1)
 
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100

# Print best guess and percent confidence
print(classes[index[0]], percentage[index[0]].item())




