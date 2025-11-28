import argparse
import json
import multiprocessing
import os
import time

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
import tifffile
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage.morphology import skeletonize

from helpers import (
    get_skeleton_lengths,
    load_tif_volume,
    pad,
    postprocess_instance,
    remove_bordering_axons,
    save_tif_volume,
    slide,
    tif_to_nii_slices,
    tif_to_tif_slices,
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(SCRIPT_DIR, "model_november_25")

# Define all directories relative to script location
split_dir = os.path.join(SCRIPT_DIR, "split")
predicted_dir = os.path.join(SCRIPT_DIR, "predicted")
assembled_dir = os.path.join(SCRIPT_DIR, "assembled")
to_predict_dir = os.path.join(SCRIPT_DIR, "to_predict")
lengths_file = os.path.join(SCRIPT_DIR, "lengths.json")

if not os.path.exists(split_dir):
    os.makedirs(split_dir)
if not os.path.exists(predicted_dir):
    os.makedirs(predicted_dir)
if not os.path.exists(assembled_dir):
    os.makedirs(assembled_dir)
if not os.path.exists(to_predict_dir):
    os.makedirs(to_predict_dir)


def draw_charts():
    completed = os.listdir(assembled_dir)
    for file in completed:
        try:
            # Skip hidden files, .gitkeep, and other system files
            if file.startswith("."):
                continue
            if "label_instances" in file:
                continue  # instances is same as binary in skeleton domain
            if file.endswith(".png"):
                continue

            file_info = file.replace(".label_binary.tif", ".tif.json")
            file_info = file_info.replace(".label_raw.tif", ".tif.json")
            print("Plotting chart for file", file)
            with open(os.path.join(to_predict_dir, file_info), "r") as finfo:
                info = json.load(finfo)
            spacing = info["scaling"]
            spacing = (spacing["z"], spacing["y"], spacing["x"])
            print("Retreived spacing:", spacing)
            in_file = os.path.join(assembled_dir, file)
            out_file = in_file + ".png"
            if os.path.exists(out_file):
                continue
            volume, _ = load_tif_volume(in_file)
            if len(volume.shape) == 4:
                volume = volume[:, -1, ...]
            print("Removing bordering axons", flush=True)
            # volume = remove_bordering_axons(volume)
            volume = remove_bordering_axons(volume)
            print("skeletonizing", flush=True)
            skeleton = skeletonize(volume)
            print("Calculating lengths", flush=True)
            lengths = get_skeleton_lengths(skeleton, spacing)
            print("Lengths calculated", flush=True)
            if os.path.exists(lengths_file):
                with open(lengths_file, "r") as length_data:
                    lengths_data = json.load(length_data)
            else:
                lengths_data = {}
            lengths_data[file] = list(lengths)
            with open(lengths_file, "w") as length_data:
                json.dump(lengths_data, length_data)

            kde = sns.kdeplot(lengths, label="KDE", color="C0")
            plt.xlabel("Length")
            plt.xticks(rotation=45)  # Rotate x-axis tick labels
            plt.legend()
            plt.xticks(rotation=45)  # Rotate x-axis tick labels
            sns.ecdfplot(lengths, ax=kde.twinx(), label="Empirical CDF", color="C1")
            plt.legend()
            plt.xticks(rotation=45)  # Rotate x-axis tick labels
            spacing_text = "Z: {:.2E} Y: {:.2E} X: {:.2E}".format(
                spacing[0], spacing[1], spacing[2]
            )
            plt.title("Label\n Spacing [m]: {}\nN axons: {},\n$\mu$: {:.2E} $\sigma$: {:.2E}".format(spacing_text, len(lengths), np.mean(lengths), np.std(lengths)))
            plt.xlabel("Length")
            plt.tight_layout()
            plt.gcf().set_dpi(300)
            plt.savefig(out_file)
            plt.close()
        except Exception as e:
            print("Error while plotting", file, e)


def merge_predictions():
    to_merge = filter(lambda x: x.endswith(".tif"), os.listdir(to_predict_dir))
    for file in to_merge:
        out_path = os.path.join(
            assembled_dir, file.replace(".tif", ".label_instances.tif")
        )
        if os.path.exists(out_path.replace("label_instances", "label_binary")):
            print("Skipping ", file, "as it's already assembled")
            continue
        file_info = file + ".json"
        with open(os.path.join(to_predict_dir, file_info), "r") as finfo:
            info = json.load(finfo)
        if len(info["shape"]) == 4:
            out = np.zeros((info["shape"][0], info["shape"][2], info["shape"][3]), bool)
        else:
            out = np.zeros(info["shape"], bool)

        for y in range(info["ny"]):
            for x in range(info["nx"]):
                patch = file.replace(".tif", "")
                patch = f"{patch}_{x}_{y}.nii.gz"
                prediction = (
                    nib.load(os.path.join(predicted_dir, patch)).get_fdata() > 0
                )
                prediction = prediction.transpose([2, 1, 0])
                # now Z Y X
                prediction = prediction[
                    :, info["pad"] : -info["pad"], info["pad"] : -info["pad"]
                ]
                print("loaded patch", patch, "with unpadded shape", prediction.shape)
                print("Patch nonzero count", np.count_nonzero(prediction))
                print("Patch mean value", np.mean(prediction))
                out[
                    :,
                    y * info["slide"] : (y + 1) * info["slide"],
                    x * info["slide"] : (x + 1) * info["slide"],
                ] = prediction

        out = np.flip(out, axis=(1, 2))
        if len(info["shape"]) == 4:
            out_temp = np.tile(out[:, np.newaxis, ...], (1, info["shape"][1], 1, 1))
            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_raw"),
                out_temp.astype(np.uint8),
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )
            del out_temp
        else:
            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_raw"),
                out.astype(np.uint8),
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )

        print("Postprocessing instance (dusting and instance segmentation)")
        out = postprocess_instance(out)

        # accomodate for additional channels if the input has them
        if len(info["shape"]) == 4:
            out = np.tile(out[:, np.newaxis, ...], (1, info["shape"][1], 1, 1))
        out = out.astype(np.uint16)
        tifffile.imwrite(
            out_path, out, compression="deflate", dtype=out.dtype, imagej=True
        )
        out = (out > 0).astype(np.uint8)
        tifffile.imwrite(
            out_path.replace(".label_instances", ".label_binary"),
            out,
            compression="deflate",
            dtype=np.uint8,
            imagej=True,
        )


def prepare_splits(ftype="tif"):
    "Loads the tif files and splits them into .nii.gz files ready for prediction"
    files = os.listdir(to_predict_dir)
    existing_files = set(os.listdir(split_dir))
    for file in files:
        if not file.endswith(".tif"):
            continue
        print("Splitting", file)
        ymax = 0
        xmax = 0
        if "nii" in ftype:
            raise NotImplementedError(
                "saving to .nii.gz needs to be updated to handle saving file info"
            )
            for nifti_slice, fname, y, x in tif_to_nii_slices(
                os.path.join(to_predict_dir, file), existing_files
            ):
                print("processing", file, "slice", y, x, "->", fname)
                nib.save(nifti_slice, os.path.join(split_dir, fname))
                xmax = max(xmax, x)
                ymax = max(ymax, y)
        elif "tif" in ftype:
            for (
                crop,
                fname,
                json_fname,
                y,
                x,
                json_content,
                shape,
                scaling,
            ) in tif_to_tif_slices(os.path.join(to_predict_dir, file), existing_files):
                print("processing", file, "slice", y, x, "->", fname)
                save_tif_volume(crop, os.path.join(split_dir, fname))
                with open(os.path.join(split_dir, json_fname), "w") as jf:
                    jf.write(json_content)
                xmax = max(xmax, x)
                ymax = max(ymax, y)
                info_file = file + ".json"
                with open(os.path.join(to_predict_dir, info_file), "w") as inf:
                    file_info = {
                        "nx": xmax + 1,
                        "ny": ymax + 1,
                        "slide": slide,
                        "pad": pad,
                        "shape": shape,
                        "scaling": scaling,
                    }
                    json.dump(file_info, inf)


def run_prediction_on_gpu(gpu_id, input_files, output_files, model_folder, n_parts):
    print(f"Starting prediction on GPU {gpu_id} with {len(input_files)} files")

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=torch.device("cuda", gpu_id),
        verbose=False,
        allow_tqdm=True,
    )

    predictor.initialize_from_trained_model_folder(
        model_folder,
        use_folds=(0, 1, 2, 3, 4),
        checkpoint_name="checkpoint_best.pth",
    )

    try:
        predictor.predict_from_files(
            input_files,
            output_files,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=2,
            num_processes_segmentation_export=2,
            folder_with_segs_from_prev_stage=None,
            part_id=0,
            num_parts=1,
        )
    except IndexError as e:
        print("Index error on prediction, most likely empty prediction target", e)
        print("Continuing...")


def predict_on_splits():
    num_gpus = torch.cuda.device_count()
    print("Running on", num_gpus, "gpus")

    input_files = os.listdir(split_dir)
    unpredicted_files = []
    unpredicted_outputs = []

    for file in input_files:
        output_name = os.path.join(predicted_dir, file.replace("_0000.tif", ".nii.gz"))
        if os.path.exists(output_name):
            continue
        if not file.endswith(".tif"):
            continue
        unpredicted_files.append([os.path.join(split_dir, file)])
        unpredicted_outputs.append(output_name)

    processes = []

    for gpu in range(num_gpus):
        p = multiprocessing.Process(
            target=run_prediction_on_gpu,
            args=(
                gpu,
                unpredicted_files[gpu::num_gpus],
                unpredicted_outputs[gpu::num_gpus],
                model_folder,
                num_gpus,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    print("All prediction jobs finished")


def main():
    parser = argparse.ArgumentParser(
        description="nnUNet inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "modes",
        nargs="*",
        help="Operation modes: split, predict, assemble, and/or analyze. If none specified, runs all modes in order.",
    )

    args = parser.parse_args()
    
    # Validate modes
    valid_modes = ["split", "predict", "assemble", "analyze"]
    if args.modes:
        for mode in args.modes:
            if mode not in valid_modes:
                parser.error(f"argument modes: invalid choice: '{mode}' (choose from {', '.join(repr(m) for m in valid_modes)})")

    # If no modes specified, run all of them
    modes_to_run = args.modes if args.modes else valid_modes

    # Run the selected modes in order
    for mode in modes_to_run:
        if mode == "split":
            start = time.time()
            print("Splitting files")
            prepare_splits()
            took = time.time() - start
            print("Splitting took", took / 60, "minutes")
        elif mode == "predict":
            print("Running prediction")
            start = time.time()
            predict_on_splits()
            took = time.time() - start
            print("Predicting took", took / 60, "minutes")
        elif mode == "assemble":
            print("Merging predictions")
            start = time.time()
            merge_predictions()
            took = time.time() - start
            print("Merging took", took / 60, "minutes")
        elif mode == "analyze":
            print("Drawing charts")
            draw_charts()


if __name__ == "__main__":
    main()
