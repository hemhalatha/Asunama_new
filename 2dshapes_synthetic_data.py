import cv2
import numpy as np
import random
import os
from PIL import Image

# Paths
backgrounds_path = "images\\bg"   # folder with natural background images
output_path = "synthetic_shapes2"
os.makedirs(output_path, exist_ok=True)

# Parameters
num_images = 10
img_size = 640
shapes = ["circle", "square", "triangle"]
shape_to_class = {"circle": 0, "square": 1, "triangle": 2}
shapes_per_image = (3, 5)  # more objects per frame
min_size, max_size = 10, 40  # smaller shapes = ~30m altitude

colors=[(255,0,0), (255,255,0), (255,165,0),(0,0,255)]

# Helper: IoU check (to avoid overlap)
def check_overlap(bbox, existing_bboxes, min_iou=0.05):
    x1, y1, x2, y2 = bbox
    for ex in existing_bboxes:
        ex1, ey1, ex2, ey2 = ex
        # Intersection
        inter_x1 = max(x1, ex1)
        inter_y1 = max(y1, ey1)
        inter_x2 = min(x2, ex2)
        inter_y2 = min(y2, ey2)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        # Areas
        area1 = (x2 - x1) * (y2 - y1)
        area2 = (ex2 - ex1) * (ey2 - ey1)
        union_area = area1 + area2 - inter_area
        iou = inter_area / union_area if union_area > 0 else 0
        if iou > min_iou:  # too much overlap
            return True
    return False

# Helper: draw shape
def draw_shape(img, shape, color, center, size, angle):
    overlay = img.copy()
    bbox = None

    if shape == "circle":
        cv2.circle(overlay, center, size, color, -1)
        bbox = [center[0]-size, center[1]-size, center[0]+size, center[1]+size]

    elif shape == "square":
        rect = np.array([
            [center[0] - size, center[1] - size],
            [center[0] + size, center[1] - size],
            [center[0] + size, center[1] + size],
            [center[0] - size, center[1] + size]
        ])
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rect = np.int32(cv2.transform(np.array([rect]), M))
        cv2.fillPoly(overlay, [rect], color)
        x_min = np.min(rect[:,:,0]); x_max = np.max(rect[:,:,0])
        y_min = np.min(rect[:,:,1]); y_max = np.max(rect[:,:,1])
        bbox = [x_min, y_min, x_max, y_max]

    elif shape == "triangle":
        pts = np.array([
            [center[0], center[1] - size],
            [center[0] - size, center[1] + size],
            [center[0] + size, center[1] + size]
        ])
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        pts = np.int32(cv2.transform(np.array([pts]), M))
        cv2.fillPoly(overlay, [pts], color)
        x_min = np.min(pts[:,:,0]); x_max = np.max(pts[:,:,0])
        y_min = np.min(pts[:,:,1]); y_max = np.max(pts[:,:,1])
        bbox = [x_min, y_min, x_max, y_max]

    return overlay, bbox

# Main loop
for i in range(num_images):
    # Background
    bg_file = random.choice(os.listdir(backgrounds_path))
    bg = Image.open(os.path.join(backgrounds_path, bg_file)).convert("RGB")
    bg = bg.resize((img_size, img_size))
    img = np.array(bg)

    annotations = []
    bboxes = []

    num_shapes = random.randint(*shapes_per_image)
    tries = 0
    while len(bboxes) < num_shapes and tries < num_shapes * 10:
        tries += 1
        shape = random.choice(shapes)
        color = random.choice(colors)
        center = (random.randint(50, img_size-50), random.randint(50, img_size-50))
        size = random.randint(min_size, max_size)
        angle = random.randint(0, 360)

        img_candidate, bbox = draw_shape(img, shape, color, center, size, angle)

        # Skip if overlap
        if bbox is None or check_overlap(bbox, bboxes):
            continue

        # Accept shape
        img = img_candidate
        bboxes.append(bbox)

        # YOLO annotation
        x_min, y_min, x_max, y_max = bbox
        x_center = ((x_min + x_max) / 2) / img_size
        y_center = ((y_min + y_max) / 2) / img_size
        w = (x_max - x_min) / img_size
        h = (y_max - y_min) / img_size
        annotations.append(f"{shape_to_class[shape]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Save image
    img_filename = f"{output_path}/synthetic_{i:04d}.jpg"
    cv2.imwrite(img_filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Save annotation
    label_filename = f"{output_path}/synthetic_{i:04d}.txt"
    with open(label_filename, "w") as f:
        f.write("\n".join(annotations))

print(f"✅ Generated {num_images} images (30m altitude, no overlap) with YOLO annotations in '{output_path}'")
