from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('DCHNet.yaml')
    model.train(data='datasets/myVisDrone.yaml', epochs=500, batch=16, imgsz=640, device=0, name='train')  # Train




