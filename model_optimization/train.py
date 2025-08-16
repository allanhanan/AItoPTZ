from ultralytics import YOLO

def main():
    # Load YOLOv8n (nano) for fast/lightweight training
    model = YOLO("yolov8n.pt")

    # Train on GPU
    model.train(
        data="model_optimization\dataset\data.yaml",        # dataset yaml
        epochs=50,                 # number of epochs
        imgsz=640,                 # image size
        device=0,                  # GPU (set to "0" for first GPU)
        workers=2,                 # some parallel workers
        project="runs",
        name="person_yolov8n_gpu"
    )

if __name__ == "__main__":
    main()
