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
    save_dir = f'full_dataset/lisa_random_patch_{args.patch_size}'
else:
    save_dir = f'nap_data/nap_topleft_patch_{args.patch_size}'
data_dir = os.path.join(save_dir, 'data')
img_dir = os.path.join(save_dir, 'image')
os.makedirs(data_dir, exist_ok=True)
os.makedirs(img_dir, exist_ok=True)

# setup model
paths.set_mode("host")
model = fasterrcnn_resnet50_fpn(pretrained=True)
art_model = PyTorchFasterRCNN(
        model=model,
        detector=None,
        clip_values=(0, 1.0),
        channels_first=False,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=None,
        attack_losses=(
            "loss_classifier",
            "loss_box_reg",
            "loss_objectness",
            "loss_rpn_box_reg",
        ),
        device_type=DEVICE,
        adaptive=False,
        defense=False,
        bpda=False,
        shape_completion=False,
        adaptive_to_shape_completion=False,
        simple_shape_completion=False,
        bpda_shape_completion=False,
        union=False
    )

#attacker = PGDPatch(art_model, batch_size=1, eps=1.0, eps_step=0.01, max_iter=200, num_random_init=0, random_eps=False,
                    #targeted=False, verbose=True)

# setup dataset

#rel_path = "./test_data/"
rel_path = "./vid/"
dataset = get_coco(rel_path, image_set="train", transforms=presets.DetectionPresetEval())


dataset_size = min(args.n_imgs, len(dataset))

print(f"dataset size: {dataset_size}")

chunk_size = dataset_size // args.world_size
start_ind = args.rank * chunk_size
print(f"start index: {start_ind}")
if args.rank == args.world_size - 1:
    end_ind = dataset_size
else:
    end_ind = (args.rank + 1) * chunk_size

print(f"end index: {end_ind}")

dataset = torch.utils.data.Subset(dataset, list(range(start_ind, end_ind)))
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1)
pbar = tqdm(data_loader)


# set up the file naming convention
# i want 6 digits. start at 1800
starting_num = 0


for i, data in enumerate(pbar):
    x, y = data
    x = x[0].unsqueeze(0).permute(0, 2, 3, 1).numpy()
    label = {k: v.squeeze(0).numpy() for k, v in y.items()}
    
    patch_height = args.patch_size
    patch_width = args.patch_size
    
    if args.random:
        h = x.shape[1]
        w = x.shape[2]
        
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
    #patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)  # Convert color channels if needed

    # Paste patch at the random position
    x_adv = x.copy()
    # convert to right color space
    x_adv[0] = cv2.cvtColor(x_adv[0], cv2.COLOR_BGR2RGB)
    x_adv[0, ymin:ymin+patch_height, xmin:xmin+patch_width, :] = patch / 255

    # Save the modified image with the naming convention
    img_fn = os.path.join(img_dir, f'{starting_num:06d}.png')
    cv2.imwrite(img_fn, (x_adv[0] * 255).astype(np.uint8))  # Ensure data type is uint8

    # Save data
    data_fn = os.path.join(data_dir, f'{starting_num:06d}.pt')
    starting_num += 1
    torch.save({
        'xmin': xmin,
        'ymin': ymin,
        'x': x,
        'y': y,
        'x_adv': x_adv,
        'patch_size': args.patch_size
    }, data_fn)
    pbar.set_description(f"{save_dir} rank {args.rank}")



    