import setuptools

setuptools.setup(
    name="dota_retraining",
    version="0.0.1a",
    author="Alexander Lay",
    packages=["dota_to_coco", "preprocess", "evaluate", "retrain", "utils"],
    install_requires=[
        "pycocotools",
        "object-detection",
        "tensorflow",
        "nms",
        "matplotlib",
        "shapely"
    ]
)
