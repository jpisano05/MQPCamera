from ultralytics import YOLO
import shutil
import os

def main():
    model = YOLO("runs/detect/train13/weights/last.pt")
    
    model.train(
        data="warpDataset/Warp-D/data.yaml",
        epochs=30,
        imgsz=640,
        batch=8,
        workers = 2,
        save_period = 5,
    )

    bestWeights = os.path.join(
        model.trainer.save_dir,
        "weights",
        "best.pt"
    )

    shutil.copy2(
        bestWeights,
        "wasteDetectorLatest.pt"
    )


if __name__ == "__main__":
    main()