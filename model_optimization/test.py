from ultralytics import YOLO
import cv2

# Load model
model = YOLO("E:/quarantine/AItoPTZ/runs/person_yolov8n_gpu/weights/best.pt")

# Run prediction (on CPU)
results = model.predict(source="E:/quarantine/AItoPTZ/model_optimization/test.png", device="cpu", conf=0.5)

# Show results manually with OpenCV and wait for key
for r in results:
    im_array = r.plot()  # plot bboxes on image
    cv2.imshow("Prediction", im_array)
    cv2.waitKey(0)       # waits indefinitely for a key press
    cv2.destroyAllWindows()
