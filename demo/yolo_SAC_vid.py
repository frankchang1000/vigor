import torch
from pathlib import Path
from PIL import Image, ImageDraw
import os
import tqdm
from patch_detector import PatchDetector
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Force reload and update YOLOv5


model = torch.hub.load("ultralytics/yolov5", "yolov5s")

vigor = PatchDetector(3, 1, base_filter=64, square_sizes=[150, 100, 75, 50, 25], n_patch=1)
vigor.unet.load_state_dict(torch.load("ckpts/CP_epoch25.pth", map_location=torch.device('cpu')))

vigor = vigor.to('cuda')

# Define COCO classes
coco_classes = [
    'person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck',
    'boat', 'trafficlight', 'firehydrant', 'streetsign', 'stopsign', 'parkingmeter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eyeglasses', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sportsball', 'kite', 'baseballbat',
    'baseballglove', 'skateboard', 'surfboard', 'tennisracket', 'bottle', 'plate',
    'wineglass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hotdog', 'pizza', 'donut', 'cake', 'chair', 'sofa',
    'pottedplant', 'bed', 'mirror', 'diningtable', 'window', 'desk', 'toilet', 'door',
    'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cellphone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors',
    'teddybear', 'hairdrier', 'toothbrush', 'hairbrush'
]

'''
# WORKING FOR ONE IMAGE
image_path = "image/000000.png"

image0 = cv2.imread(image_path)
#image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

image = np.stack([image0], axis=0).astype(np.float32)/255.0

image=image.transpose(0,3,1,2)
image=torch.tensor(image)

x_processed, _, _ = vigor(image, bpda=True, shape_completion=True)
image_sac = np.asarray(x_processed[0].cpu().detach())
image_sac =image_sac.transpose(1,2,0)

image_sac = image_sac * 255
image_sac = image_sac.astype(np.uint8)
cv2.imwrite("image_sac.png", image_sac)

# Run inference

img = Image.open("image_sac.png")
results = model(img)

# Get bounding boxes, labels, and confidence scores
boxes = results.xyxy[0][:, :4]
labels = results.xyxy[0][:, -1]
scores = results.xyxy[0][:, 4]

# Draw bounding boxes on the image
draw = ImageDraw.Draw(img)
for box, label, score in zip(boxes, labels, scores):
    if score >= 0.3:
        class_name = coco_classes[int(label)]
        draw.rectangle(xy=box.tolist(), outline="red", width=2)
        draw.text((box[0], box[1]), f"{class_name} {score:.2f}", fill="red")

# Save the image with bounding boxes
output_path  = "output/000000.png"
img.save(output_path)
print(f"Image with bounding boxes saved at: {output_path}")
'''

# multiple images
image_folder = 'image/'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
print(len(images))

for image in tqdm.tqdm(images):
    image0 = cv2.imread(os.path.join(image_folder, image))
    
    image = np.stack([image0], axis=0).astype(np.float32)/255.0

    image=image.transpose(0,3,1,2)
    image=torch.tensor(image)

    x_processed, _, _ = vigor(image, bpda=True, shape_completion=True)
    image_sac = np.asarray(x_processed[0].cpu().detach())
    image_sac =image_sac.transpose(1,2,0)

    image_sac = image_sac * 255
    image_sac = image_sac.astype(np.uint8)
    cv2.imwrite("image_sac.png", image_sac)

    # Run inference
    img = Image.open("image_sac.png")
    results = model(img)

    # Get bounding boxes, labels, and confidence scores
    boxes = results.xyxy[0][:, :4]
    labels = results.xyxy[0][:, -1]
    scores = results.xyxy[0][:, 4]

    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(img)
    for box, label, score in zip(boxes, labels, scores):
        if score >= 0.3:
            class_name = coco_classes[int(label)]
            draw.rectangle(xy=box.tolist(), outline="red", width=2)
            draw.text((box[0], box[1]), f"{class_name} {score:.2f}", fill="red")

    # Save the image with bounding boxes
    output_path  = "output/"+image
    img.save(output_path)
    print(f"Image with bounding boxes saved at: {output_path}")







