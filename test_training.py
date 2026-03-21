# smoke_test_model.py
import torch
from pathlib import Path
from ultralytics.nn.tasks import RoofModel
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)

def test_trainer():
    from ultralytics.models.yolo.roof import RoofTrainer
    overrides = {
        "data": "datasets/roof_dataset.yaml",
        "model": "yolo11n-roof.yaml",
        "epochs": 300,
        "batch": 64,
    }
    trainer = RoofTrainer(overrides=overrides)
    trainer.train()

def test_model_tunner():
    from ultralytics import YOLO
    model = YOLO("yolo11n-roof.yaml")
    customspace = { "roofcls": (1.0,34.0),
                    "roofinclination": (0.1,10.0),
                    "rooforientation": (0.1,10.0),
                    "roofarea": (0.1,10.0),
                    "classificationweight":(0.1,2)}
  
    model.tune(
        data="datasets/roof_dataset.yaml",
        epochs=200,
        task="roof",
        iterations=300,
        plots=False,
        save=False,
        val=False,
        space = customspace
    )

if __name__ == "__main__":
    print(f"Is cuda available for training?: {torch.cuda.is_available()}")
    test_model_tunner()
    test_trainer()
