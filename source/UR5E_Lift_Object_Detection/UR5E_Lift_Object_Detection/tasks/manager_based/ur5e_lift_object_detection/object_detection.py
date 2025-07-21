import torch
from ultralytics import YOLO
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

# Global variable to store the loaded model
_model = None

def load_model():
    global _model
    if _model is None:
        _model = YOLO("/home/ubuntu/Desktop/yolo/runs/detect/train/weights/best.pt")
    return _model

def inference(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera"),
) -> torch.Tensor:

    # Set the device (use GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Obtaining the bounding boxes
    camera = env.scene[camera_cfg.name]
    images = camera.data.output["rgb"]
    images = images.permute(0, 3, 1, 2).to(torch.float32).to(device) / 255.0

    model = load_model()
    model.to(device)
    results = model.predict(source=images, verbose=False)

    bounding_boxes = torch.zeros(len(images), 4, device=device)
    for i in range(len(results)):
        if len(results[i].boxes.xywhn) > 0:
            boxesLocations = torch.tensor(results[i].boxes.xywhn[0], device=device)  # Take the first bounding box only
            bounding_boxes[i] = boxesLocations

    print("\n\nResults for Env 1:")
    print(f"Number of detected objects: {len(results[0].boxes.xywhn)}")
    if len(results[0].boxes.xywhn) > 0:
        print(f"Class of Best: {results[0].boxes.cls[0]}")
        print(f"Confidence of Best: {results[0].boxes.conf[0]}")
        print(f"Bounding Box of Best: {bounding_boxes[0]}")
    else:
        print("No objects detected.")

    # Depth estimation
    depth_data = camera.data.output["depth"]
    depth_data = depth_data.permute(0, 3, 1, 2).to(torch.float32).to(device)
    print(f"Shape of Depth Image: {depth_data.shape}")

    object_distances = []
    for i in range(len(bounding_boxes)):
        x_center, y_center, width, height = bounding_boxes[i]
        x_min = int((x_center - width / 2) * depth_data.shape[3])
        x_max = int((x_center + width / 2) * depth_data.shape[3])
        y_min = int((y_center - height / 2) * depth_data.shape[2])
        y_max = int((y_center + height / 2) * depth_data.shape[2])

        # Clamp values to ensure they are within image bounds
        x_min = max(0, x_min)
        x_max = min(depth_data.shape[3] - 1, x_max)
        y_min = max(0, y_min)
        y_max = min(depth_data.shape[2] - 1, y_max)

        # Extract the depth values for the bounding box region
        depth_region = depth_data[i, 0, y_min:y_max, x_min:x_max]
        if depth_region.numel() > 0:  # Ensure the region is not empty
            avg_depth = depth_region.mean().item()
            object_distances.append(avg_depth)
        else:
            object_distances.append(0.8)  # Handle empty regions

    if len(object_distances) > 0:
        print(f"Object Distance: {object_distances[0]}")

    object_distances = torch.tensor(object_distances, device=device)

    combined_results = torch.cat((bounding_boxes, object_distances.unsqueeze(1)), dim=1)
    return combined_results