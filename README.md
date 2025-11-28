# ais_segmentation
nnUNet-based framework for segmenting axon initial segments in LSFM microscopy images

## Installation

Follow these steps to install the package and its dependencies:

### 1. Clone the repository

```bash
git clone https://github.com/HelmholtzAI-Consultants-Munich/ais_segmentation.git
cd ais_segmentation
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Download the pre-trained model

Download the `model.tar.gz` file and place it in the main project directory, then extract it:

```bash
# After downloading model.tar.gz to the ais_segmentation directory
tar -xzf model.tar.gz
```

### 4. Install the package

```bash
pip install -e .
```
#### Warning!
It is important to use the `-e` installation option in order for the script to be able to locate the model and inference files.

This will install all required dependencies and make the `nnunet_run_inference` command available.

### 5. Verify installation

```bash
nnunet_run_inference --help
```

## Usage

The `nnunet_run_inference` command supports multiple operation modes:

```bash
# Run all steps (split → predict → assemble → analyze)
nnunet_run_inference

# Run specific steps
nnunet_run_inference split
nnunet_run_inference split predict
nnunet_run_inference predict assemble analyze
```

Available modes:
- `split` - Split input data for processing
- `predict` - Run nnUNet prediction
- `assemble` - Assemble prediction results
- `analyze` - Analyze segmentation results

