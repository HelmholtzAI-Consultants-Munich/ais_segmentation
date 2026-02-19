from setuptools import setup, find_packages
from pathlib import Path

nnunet_path = Path(__file__).parent / "nnUNet"

setup(
    name="ais_segmentation",
    version="0.1.0",
    description="AIS Segmentation with nnUNet",
    author="Karol Szustakowski, Helmholtz Munich AI Consultants Team",
    author_email="karol.szustakowski@helmholtz-munich.de",
    packages=find_packages(),
    py_modules=["nnunet_run_inference", "helpers"],
    install_requires=[
        "connected-components-3d",
        "networkx",
        "Pillow",
        "skan",
        "torch==2.0.1",
        f"nnunetv2 @ file://{nnunet_path.resolve()}",
        "zarr",
        "dask",
        "lxml",
    ],
    entry_points={
        "console_scripts": [
            "nnunet_run_inference=nnunet_run_inference:main",
        ],
    },
    python_requires=">=3.9",
)
