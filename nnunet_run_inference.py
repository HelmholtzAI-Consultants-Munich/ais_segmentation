import argparse
import json
import multiprocessing
import concurrent.futures
import os
import re
import time

import dask.array as da
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
import tifffile
import torch
import zarr
import lxml
import lxml.etree
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from skimage.morphology import skeletonize
from concurrent.futures import ThreadPoolExecutor, as_completed
from dask.diagnostics import ProgressBar
from tqdm import tqdm

from helpers import (
    find_scaling,
    get_skeleton_summaries,
    load_tif_volume,
    pad,
    postprocess_instance,
    remove_bordering_axons,
    save_tif_volume,
    slide,
    counts_chunked_uint16,
    cc_gather_stats,
    stats_from_uint16_histogram,
)


def draw_charts(to_predict_dir, assembled_dir):
    to_draw = filter(
        lambda x: re.match(r".*\.label\_instances\.tiff?$", x.lower()),
        os.listdir(assembled_dir),
    )

    for file in to_draw:
        file_stub = re.sub(r"\.label\_instances\.tiff?$", "", file, flags=re.IGNORECASE)

        info_files = list(
            filter(
                lambda x: re.match(
                    rf"{re.escape(file_stub)}.*\.json$", x, flags=re.IGNORECASE
                ),
                os.listdir(to_predict_dir),
            )
        )

        if len(info_files) == 0:
            print("No info file found for", file)
            continue

        file_info = list(info_files)[0]
        print("Found info file", file_info, "for", file)

        source_file = re.sub(r".json$", "", file_info, flags=re.IGNORECASE)

        sft = tifffile.TiffFile(os.path.join(to_predict_dir, source_file))
        zarr_store = sft.aszarr()
        z = zarr.open(zarr_store, mode="r")
        dask_array = da.from_zarr(z)

        try:
            print("Plotting chart for file", file)
            with open(os.path.join(to_predict_dir, file_info), "r") as finfo:
                info = json.load(finfo)
            spacing = info["scaling"]
            transposed = info.get("transposed", False)
            print("-->", spacing)
            spacing = (spacing["z"], spacing["y"], spacing["x"])

            if transposed:
                dask_array = dask_array.transpose([1, 0, 2, 3])
            if len(dask_array.shape) == 4:
                dask_array = dask_array[:, -1, ...]

            print("Retrieved spacing:", spacing)
            in_file = os.path.join(assembled_dir, file)
            out_file = in_file + ".png"
            out_meta_file = in_file + ".axons.json"

            if os.path.exists(out_file):
                print("Skipping", file, "as it's already plotted")
                continue

            volume, _ = load_tif_volume(in_file)

            if transposed and len(volume.shape) == 4:
                volume = volume.transpose([1, 0, 2, 3])
            if len(volume.shape) == 4:
                volume = volume[:, -1, ...]

            print("Gathering stats...")
            lengths_info = cc_gather_stats(
                volume, dask_array, spacing
            )  # precompute the stats for all axons in parallel with dask
            print("Stats gathered")

            with open(out_meta_file, "w") as length_data:
                json.dump(lengths_info, length_data, indent=2)

            lengths = [info["length"] for info in lengths_info.values()]

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
            plt.title(
                "Label\n Spacing [m]: {}\nN axons: {},\nμ: {:.2E} σ: {:.2E}".format(
                    spacing_text, len(lengths), np.mean(lengths), np.std(lengths)
                )
            )
            plt.xlabel("Length")
            plt.tight_layout()
            plt.gcf().set_dpi(300)
            plt.savefig(out_file)
            plt.close()
        except Exception as e:
            print("Error while plotting", file, e)
            raise e


def load_prediction_to_volume(patch_path, out, info, y, x):
    prediction = nib.load(patch_path).get_fdata() > 0
    prediction[prediction > 0] = 0xFF
    prediction = prediction.transpose([2, 1, 0])
    # now Z Y X
    prediction = prediction[:, info["pad"] : -info["pad"], info["pad"] : -info["pad"]]
    print("loaded patch", patch_path, "with unpadded shape", prediction.shape)
    print("Patch nonzero count", np.count_nonzero(prediction))
    print("Patch mean value", np.mean(prediction))
    out[
        :,
        y * info["slide"] : (y + 1) * info["slide"],
        x * info["slide"] : (x + 1) * info["slide"],
    ] = prediction


def merge_predictions(to_predict_dir, predicted_dir, assembled_dir):
    to_merge = filter(
        lambda x: re.match(r".*\.tiff?$", x.lower()), os.listdir(to_predict_dir)
    )
    for file in to_merge:
        out_path = os.path.join(
            assembled_dir,
            re.sub(r"\.tiff?$", ".label_instances.tif", file, flags=re.IGNORECASE),
        )
        if os.path.exists(out_path.replace("label_instances", "label_binary")):
            print("Skipping ", file, "as it's already assembled")
            continue
        file_info = file + ".json"
        with open(os.path.join(to_predict_dir, file_info), "r") as finfo:
            info = json.load(finfo)
        if len(info["shape"]) == 4:
            first_axis = (
                info["shape"][0]
                if not info.get("transposed", False)
                else info["shape"][1]
            )
            out = np.zeros((first_axis, info["shape"][2], info["shape"][3]), np.uint8)
        else:
            out = np.zeros(info["shape"], np.uint8)

        def load_patch(args):
            y, x = args
            patch = file + f".part_{x}_{y}_0000.nii.gz"
            load_prediction_to_volume(
                os.path.join(predicted_dir, patch), out, info, y, x
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=26) as executor:
            futures = []
            for y in range(info["ny"]):
                for x in range(info["nx"]):
                    futures.append(executor.submit(load_patch, (y, x)))
            concurrent.futures.wait(futures)

        print("All patches merged for file", file, "with shape", out.shape)

        transposed = info.get("transposed", False)

        out = np.flip(out, axis=(1, 2))

        if len(info["shape"]) == 4:
            tile_cnt = info["shape"][0] if transposed else info["shape"][1]

            # Create broadcasted view (NO copy)
            out_b = np.broadcast_to(
                out[:, None, :, :],
                (out.shape[0], tile_cnt, out.shape[1], out.shape[2]),
            )

            if transposed:
                out_b = out_b.transpose(1, 0, 2, 3)

            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_raw"),
                out_b,
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )

            del out_b
        else:
            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_raw"),
                out,
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )

        print("Postprocessing instance (dusting and instance segmentation)")
        out = postprocess_instance(out)

        if out.dtype != np.uint16 and out.dtype != np.uint8:
            print("Warning: Postprocessed dtype is", out.dtype)
            print(
                "This should not happen and will increase memory consumption. Please report this issue."
            )
            out = out.astype(np.uint16)

        if len(info["shape"]) == 4:
            tile_cnt = info["shape"][0] if transposed else info["shape"][1]

            # Create broadcasted view (NO copy)
            out_b = np.broadcast_to(
                out[:, None, :, :],
                (out.shape[0], tile_cnt, out.shape[1], out.shape[2]),
            )

            if transposed:
                out_b = out_b.transpose(1, 0, 2, 3)

            tifffile.imwrite(
                out_path, out_b, compression="deflate", dtype=out.dtype, imagej=True
            )

            del out_b
        else:
            tifffile.imwrite(
                out_path, out, compression="deflate", dtype=out.dtype, imagej=True
            )

        binary = np.empty_like(out, dtype=np.uint8)
        np.greater(out, 0, out=binary)
        del out
        binary[binary > 0] = 0xFF

        if len(info["shape"]) == 4:
            tile_cnt = info["shape"][0] if transposed else info["shape"][1]

            # Create broadcasted view (NO copy)
            out_b = np.broadcast_to(
                binary[:, None, :, :],
                (binary.shape[0], tile_cnt, binary.shape[1], binary.shape[2]),
            )

            if transposed:
                out_b = out_b.transpose(1, 0, 2, 3)

            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_binary"),
                out_b,
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )

        else:
            tifffile.imwrite(
                out_path.replace(".label_instances", ".label_binary"),
                binary,
                compression="deflate",
                dtype=np.uint8,
                imagej=True,
            )


def get_malformed_xml(tif):
    try:
        xml = tif.pages[0].tags["ImageDescription"].value
        parser = lxml.etree.XMLParser(
            recover=True
        )  # recovers from some malformed constructs
        root = lxml.etree.fromstring(xml.encode("utf-8"), parser=parser)
        xml_fixed = lxml.etree.tostring(root, encoding="utf-8").decode("utf-8")
        return xml_fixed
    except Exception as e:
        print("Error parsing XML:", e)
        return ""


def prepare_splits(to_predict_dir, split_files_dir):
    files = os.listdir(to_predict_dir)

    existing_files = set(os.listdir(split_files_dir))

    for file in files:
        if file.lower().endswith(".tif") or file.lower().endswith(".tiff"):
            # Check if summary JSON file exists
            json_summary_path = os.path.join(to_predict_dir, file + ".json")
            if os.path.exists(json_summary_path):
                # Load the JSON to get expected split counts
                with open(json_summary_path, "r") as inf:
                    file_info = json.load(inf)
                    expected_nx = file_info.get("nx")
                    expected_ny = file_info.get("ny")

                # Check if all expected splits exist
                if expected_nx is not None and expected_ny is not None:
                    all_splits_exist = True
                    for y_idx in range(expected_ny):
                        for x_idx in range(expected_nx):
                            split_file = file + f".part_{x_idx}_{y_idx}_0000.tif"
                            split_json = file + f".part_{x_idx}_{y_idx}_0000.json"
                            if (
                                split_file not in existing_files
                                or split_json not in existing_files
                            ):
                                all_splits_exist = False
                                break
                        if not all_splits_exist:
                            break

                    if all_splits_exist:
                        print(
                            f"Skipping {file} - all splits and JSON files already exist"
                        )
                        continue

            # process the tif file
            print("Splitting", file, "detected as TIFF file")
            ymax = 0
            xmax = 0

            with tifffile.TiffFile(os.path.join(to_predict_dir, file)) as tif:
                if tif.is_imagej:
                    info = tif.imagej_metadata.get("Info", None)
                elif tif.is_ome:
                    info = getattr(tif, "ome_metadata", None)
                else:
                    print("WARNING! Unknown TIFF format for file:", file)
                    print("Metadata WILL NOT be extracted correctly.")
                    info = get_malformed_xml(tif)

                print("Converting tif to zarr")
                zarr_store = tif.aszarr()

                print("Opening zarr store")
                z = zarr.open(zarr_store, mode="r")

                shape = z.shape
                print("Building dask array")
                dask_array = da.from_zarr(z)

                transposed = False

                if len(z.shape) == 4:
                    print("Input volume has 4 dimensions")
                    if shape[0] < shape[1]:
                        print(
                            "Dimension 0 is smaller than 1, assuming channels in dim 0 and transposing to ZCYX"
                        )
                        dask_array = dask_array.transpose([1, 0, 2, 3])
                        transposed = True

                print("Original shape:", dask_array.shape)
                if len(z.shape) == 4:
                    dask_array = dask_array[:, -1, ...]
                print("Shape after removing channels:", dask_array.shape)

                print("Computing min and max values for scaling")
                min_val, perc975, max_val = stats_from_uint16_histogram(dask_array)
                print("Quantiles computed")
                print(
                    "97.5 percentile value:",
                    perc975,
                    "min value:",
                    min_val,
                    "max value:",
                    max_val,
                )

                if max_val > 255:
                    print(
                        "Volume max > 255. Assuming 16-bit input, converting to 8-bit for nnUNet (97.5 percentile will be used for scaling)"
                    )

                dask_array = da.flip(dask_array, axis=(1, 2))

                z_dim, y_dim, x_dim = dask_array.shape
                print("Volume dimensions (Z Y X):", z_dim, y_dim, x_dim)

                def load_and_save_slice(y_idx, y, x_idx, x):
                    new_path = file + f".part_{x_idx}_{y_idx}_0000.tif"
                    json_path = file + f".part_{x_idx}_{y_idx}_0000.json"
                    if new_path in existing_files:
                        print(
                            "Skipping existing split file",
                            new_path,
                        )
                        return

                    x_start = x - pad
                    x_end = x + pad + slide
                    y_start = y - pad
                    y_end = y + pad + slide

                    x_start_array, y_start_array = max(x_start, 0), max(y_start, 0)
                    x_end_array, y_end_array = min(x_end, x_dim), min(y_end, y_dim)

                    crop = dask_array[
                        :, y_start_array:y_end_array, x_start_array:x_end_array
                    ].compute()

                    if max_val > 255:
                        # use 97.5 percentile for scaling to reduce the impact of outliers
                        scale = 256.0 / (perc975 - min_val + 1)
                        crop = (crop - min_val) * scale + 0.5
                        crop = np.clip(crop, 0, 255).astype(np.uint8)

                    x_pad_start = max(x_start_array - x_start, 0)
                    y_pad_start = max(y_start_array - y_start, 0)

                    x_pad_end = min(x_end - x_end_array, pad)
                    y_pad_end = min(y_end - y_end_array, pad)

                    if y_pad_end + y_pad_start + x_pad_start + x_pad_end > 0:
                        crop = np.pad(
                            crop,
                            (
                                (0, 0),
                                (y_pad_start, y_pad_end),
                                (x_pad_start, x_pad_end),
                            ),
                            mode="constant",
                        )

                    print(
                        f"\tExtracted slice [:, {y_start_array}:{y_end_array}, {x_start_array}:{x_end_array}]"
                    )
                    print("\tAdditional padding:")
                    print(f"\t[:,{y_pad_start}:{y_pad_end}, {x_pad_start}:{x_pad_end}]")
                    save_tif_volume(crop, os.path.join(split_files_dir, new_path))

                    with open(os.path.join(split_files_dir, json_path), "w") as inf:
                        inf.write('{"spacing": [1,1,1]}')

                print("Starting multithreaded slicing and saving of patches")
                with ThreadPoolExecutor() as executor:
                    futures = []
                    for y_idx, y in enumerate(range(0, y_dim, slide)):
                        for x_idx, x in enumerate(range(0, x_dim, slide)):
                            futures.append(
                                executor.submit(load_and_save_slice, y_idx, y, x_idx, x)
                            )

                    n = len(futures)

                    for _ in tqdm(
                        as_completed(futures), total=n, desc="Multithreaded slicing"
                    ):
                        pass

                with open(os.path.join(to_predict_dir, file + ".json"), "w") as inf:
                    spacing = find_scaling(info)

                    file_info = {
                        "nx": x_idx + 1,
                        "ny": y_idx + 1,
                        "slide": slide,
                        "pad": pad,
                        "shape": shape,
                        "scaling": {**spacing},
                        "transposed": transposed,
                    }

                    json.dump(file_info, inf)


def run_prediction_on_gpu(gpu_id, input_files, output_files, model_folder, n_parts):
    print(f"Starting prediction on GPU {gpu_id} with {len(input_files)} files")
    if len(input_files) == 0:
        print(f"No files to predict on GPU {gpu_id}, skipping")
        return

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
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
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1,
            folder_with_segs_from_prev_stage=None,
            part_id=0,
            num_parts=1,
        )
    except IndexError as e:
        print("Index error on prediction, most likely empty prediction target", e)
        print("Continuing...")


def predict_on_splits(split_dir, predicted_dir, model_dir):
    num_gpus = torch.cuda.device_count()
    print("Running on", num_gpus, "gpus")

    input_files = os.listdir(split_dir)
    unpredicted_files = []
    unpredicted_outputs = []

    for file in input_files:
        output_name = os.path.join(
            predicted_dir, re.sub("_0000.tif(f)?$", "_0000.nii.gz", file)
        )
        if os.path.exists(output_name):
            continue
        if not file.endswith(".tif") and not file.endswith(".tiff"):
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
                model_dir,
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

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Directory with input files to predict on.",
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Directory with nnUNet model.",
    )

    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Directory to store results. If not specified, creates a new directory in the source directory.",
    )

    args = parser.parse_args()

    if args.results is None:
        args.results = os.path.join(args.source, "results")

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if not os.path.exists(os.path.join(args.results, "predicted")):
        os.makedirs(os.path.join(args.results, "predicted"))
    if not os.path.exists(os.path.join(args.results, "assembled")):
        os.makedirs(os.path.join(args.results, "assembled"))
    if not os.path.exists(os.path.join(args.results, "split")):
        os.makedirs(os.path.join(args.results, "split"))

    # Validate modes
    valid_modes = ["split", "predict", "assemble", "analyze"]
    if args.modes:
        for mode in args.modes:
            if mode not in valid_modes:
                parser.error(
                    f"argument modes: invalid choice: '{mode}' (choose from {', '.join(repr(m) for m in valid_modes)})"
                )

    # If no modes specified, run all of them
    modes_to_run = args.modes if args.modes else valid_modes

    # Run the selected modes in order
    for mode in modes_to_run:
        if mode == "split":
            start = time.time()
            print("Splitting files")
            prepare_splits(args.source, os.path.join(args.results, "split"))
            took = time.time() - start
            print("Splitting took", took / 60, "minutes")
        elif mode == "predict":
            print("Running prediction")
            start = time.time()
            predict_on_splits(
                os.path.join(args.results, "split"),
                os.path.join(args.results, "predicted"),
                args.model,
            )
            took = time.time() - start
            print("Predicting took", took / 60, "minutes")
        elif mode == "assemble":
            print("Merging predictions")
            start = time.time()
            merge_predictions(
                args.source,
                os.path.join(args.results, "predicted"),
                os.path.join(args.results, "assembled"),
            )
            took = time.time() - start
            print("Merging took", took / 60, "minutes")
        elif mode == "analyze":
            print("Drawing charts")
            draw_charts(args.source, os.path.join(args.results, "assembled"))


if __name__ == "__main__":
    main()
