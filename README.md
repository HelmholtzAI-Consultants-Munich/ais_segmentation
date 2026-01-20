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
pip install .
```

This will install all required dependencies and make the `nnunet_run_inference` command available.

### 5. Verify installation

```bash
nnunet_run_inference --help
```

## Usage

The `nnunet_run_inference` command supports multiple operation modes that process LSFM microscopy images through a complete segmentation pipeline.

### Command Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--source` | Yes | Directory containing input `.tif`/`.tiff` files to process |
| `--model` | Yes | Directory containing the nnUNet model weights |
| `--results` | No | Output directory (defaults to `{source}/results`) |
| `modes` | No | One or more of: `split`, `predict`, `assemble`, `analyze` |

### Pipeline Stages

The inference pipeline consists of four stages that can be run individually or all together:

```bash
# Run all steps (split → predict → assemble → analyze)
nnunet_run_inference --source /path/to/images --model /path/to/model

# Run specific steps
nnunet_run_inference --source /path/to/images --model /path/to/model split
nnunet_run_inference --source /path/to/images --model /path/to/model split predict
nnunet_run_inference --source /path/to/images --model /path/to/model predict assemble analyze

# Specify custom output directory
nnunet_run_inference --source /path/to/images --model /path/to/model --results /path/to/output
```

#### Available Modes:

**1. `split` - Data Preparation**
- Reads `.tif`/`.tiff` files from source directory
- Splits large images into smaller patches with overlap for efficient processing
- Saves patches to `results/split/` directory as `.tif` files
- Creates metadata JSON files with information about patch positions, padding, and image properties
- This step is necessary because nnUNet processes fixed-size patches

**2. `predict` - Neural Network Inference**
- Loads split patches from `results/split/` directory
- Runs nnUNet deep learning model on each patch using available GPUs
- Performs segmentation to identify axon initial segments
- Saves predictions to `results/predicted/` directory as `.nii.gz` files
- Automatically distributes workload across multiple GPUs if available

**3. `assemble` - Result Reconstruction**
- Reads predictions from `results/predicted/` directory
- Reconstructs full-size images by stitching patches back together
- Applies post-processing:
  - Instance segmentation to separate individual axons
  - Dust removal to clean up noise
- Saves three types of outputs to `results/assembled/` directory:
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
  - `.axons.json` file per image containing detailed measurements for each axon (length, volume, mean intensity)

### Directory Structure

After running the pipeline, your directory structure will look like this:

```
source_directory/
├── image1.tif              # Your input files
├── image2.tiff
├── image1.tif.json         # Auto-generated metadata
├── image2.tiff.json
└── results/                # Default output directory (or custom --results path)
    ├── split/              # Temporary: Split patches
    ├── predicted/          # Temporary: Raw predictions (.nii.gz)
    └── assembled/          # Output: Final results
        ├── *.label_binary.tif
        ├── *.label_instances.tif
        ├── *.label_raw.tif
        ├── *.label_instances.tif.png
        └── *.label_instances.tif.axons.json
```

### Example Workflow

```bash
# 1. Run complete pipeline on your images
nnunet_run_inference --source /path/to/my/images --model /path/to/model_november_25

# 2. Find results in /path/to/my/images/results/assembled/
```

### Output JSON Format

The `analyze` step generates an `.axons.json` file for each processed image containing detailed measurements for every detected axon. The file is a dictionary where keys are instance IDs (matching the labels in `.label_instances.tif`):

```json
{
  "1": {
    "length": 0.000234,
    "volume": 15823,
    "profile": [128, 135, 142, 138, ...]
  },
  "2": {
    "length": 0.000189,
    "volume": 12456,
    "profile": [98, 102, 107, 104, ...]
  }
}
```

| Field | Description |
|-------|-------------|
| `length` | Length of the axon's skeleton path (in meters, using image spacing metadata) |
| `volume` | Number of voxels belonging to this axon instance |
| `profile` | Intensity values sampled along the skeleton path from the original image |

### Tips

- The pipeline remembers progress and skips already-processed files
- You can run stages individually if you need to interrupt and resume
- GPU memory usage can be high; the pipeline uses all available GPUs automatically
- Processing time varies with image size and number of GPUs (typically minutes to hours)


