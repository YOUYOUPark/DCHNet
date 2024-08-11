from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('runs/detect/train/weights/best.pt')
    model.val(data='datasets/myVisDrone.yaml', imgsz=640, batch=16, device=0, split='test', name='val')  # Test

