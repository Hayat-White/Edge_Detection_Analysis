from ultralytics import YOLO
from pycocotools.coco import COCO
import os
import json
from tqdm import tqdm
import itertools

# Path setup
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "coco2017_canny_images")
image_dir = os.path.join(dataset_dir, "val2017")
ann_path = os.path.join(dataset_dir, "annotations", "instances_val2017.json")
model_folders = [
    "Basic_trained_model",
    "canny_trained_model",
    "Laplacian_trained_model",
    "Sobel_trained_model",
    "Prewitt_trained_model"
]

# Load COCO ground truth
coco = COCO(ann_path)

# Pick 1000 image IDs for evaluation
image_ids = coco.getImgIds()[:5000]
images = coco.loadImgs(image_ids)


total_gt_objects = 0
for img in images:
    ann_ids = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    total_gt_objects += len(anns)

print(f"\n Ground truth total objects in 5000 images: {total_gt_objects}\n")

results = {}

for model_folder in model_folders:
    model_path = os.path.join(model_folder, "best.pt")
    model = YOLO(model_path)

    print(f"🔍 Evaluating: {model_folder}")
    total_pred_objects = 0

    for img_meta in tqdm(images):
        img_path = os.path.join(image_dir, img_meta['file_name'])
        pred = model(img_path, verbose=False)[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy()
        total_pred_objects += len(pred_boxes)

    detection_ratio = total_pred_objects / total_gt_objects
    results[model_folder] = {
        "predicted_objects": total_pred_objects,
        "detection_ratio": round(detection_ratio, 4)
    }

# === PRINT RESULTS ===
print("\n Detection Comparison Results:")
print(f"{'Model':30} | {'Predicted':>10} | {'Ratio vs GT':>12}")
print("-" * 60)
for model, stats in results.items():
    print(f"{model:30} | {stats['predicted_objects']:10} | {stats['detection_ratio']:12.4f}")




for modelfolder in model_folders:
    model_path = os.path.join(modelfolder, "best.pt")
    model = YOLO(model_path)

    print(f"Evaluating: {modelfolder}")
    total_pred_objects = 0

    for img_meta in tqdm(images):
        img_path = os.path.join(image_dir, img_meta['file_name'])
        pred = model(img_path, verbose=False)[0]
        pred_boxes = pred.boxes.xyxy.cpu().numpy()
        total_pred_objects += len(pred_boxes)

    results[modelfolder] = total_pred_objects

# === PRINT TOTAL DETECTIONS ===
print("\n Total Detections by Model:")
for model, count in results.items():
    print(f"{model:30}: {count} objects")

# === PAIRWISE COMPARISONS ===
print("\n Pairwise Detection Differences:")
pairs = list(itertools.combinations(model_folders, 2))

for m1, m2 in pairs:
    c1, c2 = results[m1], results[m2]
    diff = c1 - c2
    percent = (diff / c2) * 100 if c2 != 0 else float('inf')

    print(f"{m1:25} vs {m2:25} --> difference: {diff:+6}  ({percent:+6.2f}%)")



# Loop over each model
# for model_folder in model_folders:
#     print(f"\nEvaluating model: {model_folder}")
#     model_path = os.path.join(model_folder, "best.pt")
#     model = YOLO(model_path)

#     total_gt = 0
#     total_pred = 0
#     correct = 0

#     for img_meta in tqdm(images):
#         img_path = os.path.join(image_dir, img_meta['file_name'])

#         # Run inference
#         results = model(img_path)[0]
#         pred_boxes = results.boxes.xyxy.cpu().numpy()

#         # Get ground truth boxes
#         ann_ids = coco.getAnnIds(imgIds=img_meta['id'])
#         anns = coco.loadAnns(ann_ids)
#         gt_boxes = [ann['bbox'] for ann in anns if ann['iscrowd'] == 0]

#         total_gt += len(gt_boxes)
#         total_pred += len(pred_boxes)

#         # (Optional) Do IoU-based matching for more accurate comparison
#         # You can implement IoU matching to count how many predicted boxes are correct

#     print(f"Total GT objects: {total_gt}")
#     print(f"Total predicted objects: {total_pred}")
#     print(f"Difference: {abs(total_gt - total_pred)}\n")
