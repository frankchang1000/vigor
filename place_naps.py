# same as generate_adv_data_coco.py, except for naturalistic adversarial patches

from armory import paths
from vision.torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from pytorch_faster_rcnn import PyTorchFasterRCNN
from torchvision.datasets import CocoDetection
from tqdm import tqdm
from coco_utils import get_coco
import presets
import torch
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser(description="Generating Naturalistic Adversarial Patch Dataset")
parser.add_argument("--world_size", type=int, default=1, help='total number of jobs')
parser.add_argument("--rank", type=int, default=0, help='job ID')
parser.add_argument("--patch_size", type=int, default=100)
parser.add_argument("--random", action='store_true', default=True)
parser.add_argument("--device", type=str, default='0')
parser.add_argument("--n_imgs", type=int, default=450, help='number of adv images to generate')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# setup dir
if args.random:
    save_dir = f'lisa_demo/lisa_random_patch_{args.patch_size}'
else:
    save_dir = f'lisa_demo/nap_topleft_patch_{args.patch_size}'
data_dir = os.path.join(save_dir, 'data')
img_dir = os.path.join(save_dir, 'image')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)


#rel_path = "./test_data/"
#rel_path = "./vid/"
# Specify the path to the folder containing images
image_folder_path = "./vid/train2017/"

# Get a list of image file names in the folder
image_file_names = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

print(f"Found {len(image_file_names)} images in the specified folder.")
# Use the specified image folder instead of COCO dataset
dataset = [(os.path.join(image_folder_path, image_file), {}) for image_file in image_file_names]

dataset_size = min(args.n_imgs, len(dataset))

# ...

# Iterate through the images in the specified folder
for i in tqdm(range(dataset_size)):
    image_file, _ = dataset[i]
    x = cv2.imread(image_file)  # Load the image using OpenCV

    patch_height = args.patch_size
    patch_width = args.patch_size
    
    if args.random:
        h = x.shape[0]
        w = x.shape[1]
        
        # Randomly select the position of the top-left corner for the patch
        ymin = np.random.randint(0, h - patch_height)
        xmin = np.random.randint(0, w - patch_width)
    else:
        xmin = 0
        ymin = 0
    
    # Load patch
    patch_dir = 'patches/'
    patch_fn = np.random.choice(os.listdir(patch_dir))
    patch = cv2.imread(os.path.join(patch_dir, patch_fn))

    # Ensure patch has the same number of color channels as the original image
    patch = cv2.resize(patch, (patch_width, patch_height))
    
    # Paste patch at the random position
    x_adv = x.copy()
    x_adv[ymin:ymin+patch_height, xmin:xmin+patch_width, :] = patch

    # Save the modified image with the naming convention
    img_fn = os.path.join(img_dir, f'{i:06d}.png')
    cv2.imwrite(img_fn, x_adv)

    #pbar.set_description(f"{save_dir} rank {args.rank}")



    