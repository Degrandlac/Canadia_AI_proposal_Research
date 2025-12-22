from ultralytics import YOLO

def main():
    model = YOLO(
        r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\Baseline\weights\last.pt"
    )

    model.val(
        data=r"C:\Users\Kevin\Desktop\malaria\codes\Malaria_research\yolov8_baseline\ultralytics\journal-paper-experiment-1-4\data.yaml",
        split="test",
        workers=0  # IMPORTANT
    )

if __name__ == "__main__":
    main()
