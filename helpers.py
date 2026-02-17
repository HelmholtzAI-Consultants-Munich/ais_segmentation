import os
import xml.etree.ElementTree as ET
from os.path import basename

import cc3d
import networkx as nx
import nibabel as nib
import numpy as np
import tifffile
from PIL import Image
from skan import Skeleton, summarize

Image.MAX_IMAGE_PIXELS = 281309436 * 12

files = os.listdir(".")

slide = 2000
patch_size = 256
pad = patch_size // 2


def build_skeleton_graph(skeleton_summary):
    # skeleton_summary = skeleton_summary[skeleton_summary["branch_type"].isin([0, 1, 2])]
    return nx.from_pandas_edgelist(
        skeleton_summary, "node_id_src", "node_id_dst", edge_attr="branch_distance"
    )


def get_skeleton_summaries(skeleton_inst, original_file_dask, spacing):
    skeleton = Skeleton(skeleton_inst > 0, spacing=spacing, keep_images=False)
    stats = summarize(skeleton, separator="_")

    path_lengths = np.asarray(skeleton.path_lengths(), dtype=np.float32)

    out = {}
    entries = []
    flats = []

    for skel_id, rows in stats.groupby("skeleton_id", sort=False):
        path_ids = rows.index.to_numpy(dtype=int)

        if path_ids.size == 0:
            continue

        best_pid = int(path_ids[np.argmax(path_lengths[path_ids])])
        length = float(path_lengths[best_pid])

        coords = skeleton.path_coordinates(best_pid)

        if coords.size == 0:
            continue

        vox = np.rint(coords).astype(np.int64)
        c0 = tuple(vox[0])
        inst_id = int(skeleton_inst[c0])

        if inst_id == 0:
            print(
                "Warning: skeleton_id",
                skel_id,
                "has best path starting at background voxel",
                c0,
            )
            continue

        idx = tuple(vox.T)
        flat = np.ravel_multi_index(
            idx, dims=original_file_dask.shape, mode="clip"
        ).astype(np.int64)

        entries.append((inst_id, length, flat.size))
        flats.append(flat)

    if len(flats) == 0:
        return out

    flat = np.concatenate(flats, axis=0)
    profile = original_file_dask.ravel()[flat].compute().astype(float)

    pos = 0
    for inst_id, length, n in entries:
        prof = profile[pos : pos + n].tolist()
        pos += n

        if inst_id in out:
            prev_len = out[inst_id]["length"]
            if length > prev_len:
                out[inst_id] = {"length": length, "profile": prof}
        else:
            out[inst_id] = {"length": length, "profile": prof}

    return out


def get_skeleton_lengths(skeleton, spacing):
    skeleton = Skeleton(skeleton, spacing=spacing)
    skeleton_summary = summarize(skeleton, separator="_")
    return skeleton_summary.groupby("skeleton_id")[
        ["node_id_src", "node_id_dst", "branch_type", "branch_distance"]
    ].apply(
        lambda rows: nx.diameter(build_skeleton_graph(rows), weight="branch_distance")
    )


def remove_bordering_axons(volume, precomputed_ccl=False):
    print("Border removal - computing ccl")
    if not precomputed_ccl:
        volume = cc3d.connected_components(volume, connectivity=26, max_labels=0xFFFE, out_dtype=np.uint16)
    else:
        volume = volume

    # Efficiently gather border labels using ravel (no unnecessary copies)
    border_labels = np.unique(
        np.concatenate(
            [
                volume[0, :, :].ravel(),
                volume[-1, :, :].ravel(),
                volume[:, 0, :].ravel(),
                volume[:, -1, :].ravel(),
                volume[:, :, 0].ravel(),
                volume[:, :, -1].ravel(),
            ]
        )
    )

    # Create mask of where labels should be removed
    mask = np.isin(volume, border_labels, kind="table")

    # Set those to zero (or preserve binary structure)
    volume[mask] = 0
    return volume

def postprocess_instance(volume):
    # the smallest real example we have seen so far is around 2000 voxels big
    labels_out = cc3d.connected_components(volume, binary_image=True, connectivity=26, max_labels=0xFFFE, out_dtype=np.uint16)
    print("Number of labels before dusting:", np.amax(labels_out))
    labels_out = cc3d.dust(
        labels_out, precomputed_ccl=True, in_place=True, threshold=1500
    )
    print("Number of labels before border removal:", np.amax(labels_out))
    # labels_out = remove_bordering_axons(labels_out, precomputed_ccl=True)
    # print("Number of labels after border removal:", len(np.unique(labels_out))-1)
    return labels_out


def extract_value(line):
    line = line.split("=")[-1]
    return float(line)


def find_scaling(info):
    if info is None:
        return {"y": 1.0, "x": 1.0, "z": 1.0}

    if "PhysicalSizeX" in info:  # ome XML metadata
        root = ET.fromstring(info)
        # detect default namespace
        ns = {"ome": root.tag.split("}")[0].strip("{")}

        values = {"PhysicalSizeX": 0, "PhysicalSizeY": 0, "PhysicalSizeZ": 0}

        for elem in root.iter():
            for key in values:
                if key in elem.attrib:
                    values[key] = float(elem.attrib[key])

        return {
            "y": values["PhysicalSizeY"] * 1e-6,  # values are in micrometers
            "x": values["PhysicalSizeX"] * 1e-6,
            "z": values["PhysicalSizeZ"] * 1e-6,
        }

    elif "AcquisitionBlock" in info:  # ImageJ metadata
        info = info.split("\n")
        info = filter(
            lambda x: "AcquisitionBlock|AcquisitionModeSetup|Scaling" in x, info
        )
        info = list(info)
        scaling_y, scaling_x, scaling_z = 1.0, 1.0, 1.0
        for line in info:
            if "ScalingX" in line:
                scaling_x = extract_value(line)
            elif "ScalingY" in line:
                scaling_y = extract_value(line)
            elif "ScalingZ" in line:
                scaling_z = extract_value(line)
        return {"y": scaling_y, "x": scaling_x, "z": scaling_z}
    
    else:
        print("WARNING! Unknown TIFF format for file. Assuming isotropic spacing of 1.0.")
        return {"y": 1.0, "x": 1.0, "z": 1.0}



def load_tif_volume(path):
    volume = tifffile.imread(path)  # Reads entire stack
    # volume = volume[:, ::-1, :]  # Mirror Y
    info = None
    with tifffile.TiffFile(path) as tif:
        try:
            info = tif.imagej_metadata["Info"]
        except:
            pass
        if info is None:
            try:
                info = tif.ome_metadata["Info"]
            except:
                pass
        if info is None:
            try:
                info = tif.ome_metadata
            except:
                pass
    return volume, info


def save_tif_volume(volume, path, compression="packbits"):
    if compression:
        tifffile.imwrite(path, volume, compression=compression)
    else:
        tifffile.imwrite(path, volume)


def tif_to_tif_slices(path, existing_files, spacing=[1, 1, 1]):
    volume, info = load_tif_volume(path)

    shape = volume.shape

    transposed = False

    if len(shape) == 4:
        print("Input volume has 4 dimensions")
        if shape[0] < shape[1]:
            print(
                "Dimension 0 is smaller than 1, assuming channels in dim 0 and transposing to ZCYX"
            )
            volume = volume.transpose([1, 0, 2, 3])
            transposed = True

    volume_max = volume.max()
    if volume_max > 255:
        print("Volume max > 255. Assuming 16-bit input, converting to 8-bit for nnUNet")
        if len(shape) == 3:
            volume = volume.astype(np.float16)
            volume_min = volume.min()
            scale = 256.0 / (volume_max - volume_min + 1)
            volume = (volume - volume_min) * scale + 0.5
            volume = np.clip(volume, 0, 255).astype(np.uint8)
        elif len(shape) == 4:
            volume = volume.astype(np.float16)
            for c in range(volume.shape[1]):
                channel_min = volume[:, c].min()
                channel_max = volume[:, c].max()
                scale = 256.0 / (channel_max - channel_min + 1)
                volume[:, c] = (volume[:, c] - channel_min) * scale + 0.5
            volume = np.clip(volume, 0, 255).astype(np.uint8)

    scaling = find_scaling(info)
    orig_shape = volume.shape
    print("original shape:", orig_shape)
    if len(volume.shape) == 4:
        volume = volume[:, 1]
    print("Shape after removing channels:", volume.shape)
    volume = np.flip(volume, axis=(1, 2))

    z_dim, y_dim, x_dim = volume.shape

    for y_idx, y in enumerate(range(0, y_dim, slide)):
        for x_idx, x in enumerate(range(0, x_dim, slide)):
            new_path = (
                path.replace(".tiff", ".tif")
                .replace(".TIFF", ".tif")
                .replace(".TIF", ".tif")
            )
            fname = basename(new_path.replace(".tif", f"_{x_idx}_{y_idx}_0000.tif"))
            json_name = basename(
                new_path.replace(".tif", f"_{x_idx}_{y_idx}_0000.json")
            )
            spacing_info = (
                '{"spacing": ['
                + str(spacing[0])
                + ","
                + str(spacing[1])
                + ","
                + str(spacing[2])
                + "]}"
            )
            print("Processing slice", y, x)
            if json_name in existing_files:
                print("Slice ", y, x, "already exists in output folder. skipping.")
                continue

            x_start = x - pad
            x_end = x + pad + slide
            y_start = y - pad
            y_end = y + pad + slide

            x_start_array, y_start_array = max(x_start, 0), max(y_start, 0)
            x_end_array, y_end_array = min(x_end, x_dim), min(y_end, y_dim)

            crop = volume[:, y_start_array:y_end_array, x_start_array:x_end_array]

            x_pad_start = max(x_start_array - x_start, 0)
            y_pad_start = max(y_start_array - y_start, 0)

            x_pad_end = min(x_end - x_end_array, pad)
            y_pad_end = min(y_end - y_end_array, pad)

            if y_pad_end + y_pad_start + x_pad_start + x_pad_end > 0:
                crop = np.pad(
                    crop,
                    ((0, 0), (y_pad_start, y_pad_end), (x_pad_start, x_pad_end)),
                    mode="constant",
                )

            print(
                f"\tExtracted slice [:, {y_start_array}:{y_end_array}, {x_start_array}:{x_end_array}]"
            )
            print("\tAdditional padding:")
            print(f"\t[:,{y_pad_start}:{y_pad_end}, {x_pad_start}:{x_pad_end}]")

            yield (
                crop,
                fname,
                json_name,
                y_idx,
                x_idx,
                spacing_info,
                orig_shape,
                scaling,
                transposed,
            )


def tif_to_nii_slices(path, existing_files):
    volume = load_tif_volume(path)
    if len(volume.shape) == 4:
        volume = volume[:, 1]
    volume = volume.transpose([2, 1, 0])
    # volume = volume[:, ::-1, :]
    # volume = volume[::-1, :, :]
    volume = np.flip(volume, axis=(0, 1))
    volume = np.pad(volume, ((pad, pad), (pad, pad), (0, 0)), mode="constant")

    x_dim, y_dim, z_dim = volume.shape

    for y_idx, y in enumerate(range(pad, y_dim - pad, slide)):
        for x_idx, x in enumerate(range(pad, x_dim - pad, slide)):
            fname = basename(path.replace(".tif", f"_{x_idx}_{y_idx}_0000.nii.gz"))
            print("Processing slice", y, x)
            if fname in existing_files:
                print("Slice ", y, x, "already exists in output folder. skipping.")
                continue
            y_end = min(y + slide + pad, y_dim)
            x_end = min(x + slide + pad, x_dim)
            # crop = volume[x:x_end, y:y_end, :]
            crop = volume[x - pad : x_end, y - pad : y_end]

            nifti_image = nib.Nifti1Image(crop, np.eye(4))
            yield nifti_image, fname, y_idx, x_idx
