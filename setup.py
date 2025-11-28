from setuptools import setup, find_packages

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
        "nibabel",
        "numpy",
        "tifffile",
        "Pillow",
        "skan",
        "matplotlib",
        "seaborn",
        "torch",
        "nnunetv2",
        "scikit-image",
    ],
    entry_points={
        "console_scripts": [
            "nnunet_run_inference=nnunet_run_inference:main",
        ],
    },
    python_requires=">=3.9",
)
