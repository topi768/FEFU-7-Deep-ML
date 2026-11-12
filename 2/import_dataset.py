"""Загрузка трёх классов листьев картофеля из PlantVillage."""

from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


DATASET_DIR = Path(__file__).resolve().parent / "datasets" / "plantvillage-potato"
CLASSES = ("Potato___Early_blight", "Potato___healthy", "Potato___Late_blight")
SOURCE = "https://github.com/spMohanty/PlantVillage-Dataset.git"


def download_dataset() -> Path:
    if DATASET_DIR.exists():
        return DATASET_DIR

    DATASET_DIR.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".plantvillage-", dir=DATASET_DIR.parent) as temp:
        work = Path(temp)
        repo = work / "source"
        # Загружаем только папки трёх классов из цветной версии датасета
        subprocess.run(
            ["git", "clone", "--depth=1", "--filter=blob:none", "--no-checkout", SOURCE, str(repo)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(repo), "sparse-checkout", "set", "--cone",
             *(f"raw/color/{name}" for name in CLASSES)],
            check=True,
        )
        subprocess.run(["git", "-C", str(repo), "checkout", "HEAD"], check=True)

        images = work / "images"
        images.mkdir()
        for name in CLASSES:
            shutil.move(repo / "raw" / "color" / name, images / name)
        images.rename(DATASET_DIR)
    return DATASET_DIR


if __name__ == "__main__":
    print(download_dataset())
