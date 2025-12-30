from ultralytics import YOLO
import os

def train_yolo_obb():

    model = YOLO('yolov8n-obb.pt')

    results = model.train(
        data='data.yaml',
        epochs=100,
        imgsz=1024,
        batch=16,
        workers=12,
        device=0,
        cache=True,
        amp=True,
        project='runs/train',
        name='ship_obb_model'
    )
    
    print("Eğitim tamamlandı. Model 'runs/train/ship_obb_model' altına kaydedildi.")

if __name__ == "__main__":
    train_yolo_obb()
