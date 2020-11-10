from pathlib import Path
from shape_recognition import ShapeRecognition
import os
import cv2

images = []
possible_shapes = [
    "circle",
    "cross",
    "heptagon",
    "quartercircle",
    "rectangle",
    "semicircle",
    "star",
    "triangle"
]

def load_images():
    p = Path('../Image Dataset/Close Ups')
    for shape in os.scandir(p):
        if shape.is_dir():
            for image in os.scandir(shape):
                if Path(image).suffix == ".png":
                    images.append({
                        "path": str(Path(image).absolute()),
                        "shape": Path(image).name.split("_")[0],
                        "background_color": Path(image).name.split("_")[1],
                        "letter_color": Path(image).name.split("_")[3][:-4],
                        "letter": Path(image).name.split("_")[2],
                        "image": cv2.imread(str(Path(image).absolute()))
                    })


def main():
    load_images()
    ShapeRecognition().classify_image(images[1]["image"])

main()
