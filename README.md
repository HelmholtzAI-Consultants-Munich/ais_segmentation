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

The `nnunet_run_inference` command supports multiple operation modes that process LSFM microscopy images through a complete segmentation pipeline.

### Input Data

Place your `.tif` image files in the `to_predict/` directory within the project root. The pipeline will automatically process all `.tif` files found in this directory.

### Pipeline Stages

The inference pipeline consists of four stages that can be run individually or all together:

```bash
# Run all steps (split → predict → assemble → analyze)
nnunet_run_inference

# Run specific steps
nnunet_run_inference split
nnunet_run_inference split predict
nnunet_run_inference predict assemble analyze
```

#### Available Modes:

**1. `split` - Data Preparation**
- Reads `.tif` files from `to_predict/` directory
- Splits large images into smaller patches with overlap for efficient processing
- Saves patches to `split/` directory as `.tif` files with `_0000.tif` suffix
- Creates metadata JSON files with information about patch positions, padding, and image properties
- This step is necessary because nnUNet processes fixed-size patches

**2. `predict` - Neural Network Inference**
- Loads split patches from `split/` directory
- Runs nnUNet deep learning model on each patch using available GPUs
- Performs segmentation to identify axon initial segments
- Saves predictions to `predicted/` directory as `.nii.gz` files
- Automatically distributes workload across multiple GPUs if available

**3. `assemble` - Result Reconstruction**
- Reads predictions from `predicted/` directory
- Reconstructs full-size images by stitching patches back together
- Applies post-processing:
  - Instance segmentation to separate individual axons
  - Dust removal to clean up noise
- Saves three types of outputs to `assembled/` directory:
  - `.label_raw.tif` - Binary segmentation mask
  - `.label_instances.tif` - Instance-segmented axons (each axon has unique ID)
  - `.label_binary.tif` - Clean binary mask after post-processing

**4. `analyze` - Quantitative Analysis**
- Processes segmented axons from `assembled/` directory
- Removes axons touching image borders
- Computes skeleton representation of each axon
- Calculates axon length statistics using image spacing metadata
- Generates visualization outputs:
  - `.png` charts showing length distribution (KDE plot and empirical CDF)
  - `lengths.json` file containing all measured axon lengths

### Directory Structure

After running the pipeline, your project will have the following structure:

```
ais_segmentation/
├── to_predict/          # Input: Place your .tif files here
├── split/               # Temporary: Split patches
├── predicted/           # Temporary: Raw predictions
├── assembled/           # Output: Reconstructed segmentations
│   ├── *.label_binary.tif
│   ├── *.label_instances.tif
│   ├── *.label_raw.tif
│   └── *.label_binary.tif.png
└── lengths.json         # Output: Quantitative measurements
```

### Example Workflow

```bash
# 1. Place your images
cp /path/to/your/images/*.tif to_predict/

# 2. Run complete pipeline
nnunet_run_inference

# 3. Find results in assembled/ directory and lengths.json
```

### Tips

- The pipeline remembers progress and skips already-processed files
- You can run stages individually if you need to interrupt and resume
- GPU memory usage can be high; the pipeline uses all available GPUs automatically
- Processing time varies with image size and number of GPUs (typically minutes to hours)


