import torch
from ultralytics import YOLO

def load_model():

    model = YOLO("/home/ubuntu/Desktop/yolo/runs/detect/train/weights/best.pt")
    return model

def reset_model(model, env_ids):
    pass

def inference(model, images: torch.Tensor) -> torch.Tensor:

    images = images.permute(0, 3, 1, 2).to(torch.float32) / 255.0

    results = model.predict(source=images, verbose=False)

    bounding_boxes = torch.zeros(len(images), 4)
    for i in range(len(results)):
        if len(results[i].boxes.xywhn) > 0:
            boxesLocations = results[i].boxes.xywhn[0] # Take the first detected box in the image
            bounding_boxes[i] = boxesLocations

    print(bounding_boxes)

    return bounding_boxes