import os
import numpy as np
import nibabel as nib
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# -------- CONFIG --------

INPUT_DIR = r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\Training Batch 2"

OUTPUT_DIR = r"D:\EC22B1065\media\nas\01_Datasets\CT\LITS\2D_Slices"

SLICE_AXIS = 2          # 0=Sagittal, 1=Coronal, 2=Axial
SKIP_EMPTY_MASK_SLICES = True
HU_MIN = -200
HU_MAX = 400
RESIZE_TO = None
# ------------------------


def window_and_normalize(volume_data, hu_min, hu_max):
    # Clip HU values and scale to 0–255
    volume_data = np.clip(volume_data, hu_min, hu_max)
    volume_data = (volume_data - hu_min) / (hu_max - hu_min)
    return (volume_data * 255).astype(np.uint8)


def normalize_mask(mask_data):
    # Convert labels 0,1,2 to visible grayscale
    mask_data = mask_data.astype(np.uint8)
    out = np.zeros_like(mask_data, dtype=np.uint8)
    out[mask_data == 1] = 127
    out[mask_data == 2] = 255
    return out


def get_slice(data, idx, axis):
    # Extract one 2D slice from 3D volume
    if axis == 0:
        return data[idx, :, :]
    elif axis == 1:
        return data[:, idx, :]
    else:
        return data[:, :, idx]


def save_slice(array_2d, filepath, resize_to=None):
    img = Image.fromarray(array_2d)
    if resize_to:
        img = img.resize(resize_to, Image.LANCZOS)
    img.save(filepath)


def process_nii_file(nii_path, out_subdir, is_mask, mask_path=None):
    # Load .nii file
    nii = nib.load(str(nii_path))
    data = nii.get_fdata()

    # Apply processing depending on type
    if not is_mask:
        data = window_and_normalize(data, HU_MIN, HU_MAX)
    else:
        data = normalize_mask(data.astype(np.int16))

    n_slices = data.shape[SLICE_AXIS]

    # Load mask for filtering empty slices
    mask_data = None
    if SKIP_EMPTY_MASK_SLICES and mask_path and mask_path.exists():
        mask_data = nib.load(str(mask_path)).get_fdata().astype(np.int16)

    saved_count = 0
    for i in range(n_slices):
        slice_2d = get_slice(data, i, SLICE_AXIS)

        # Skip slices with no liver/tumor
        if SKIP_EMPTY_MASK_SLICES:
            if mask_data is not None:
                mask_slice = get_slice(mask_data, i, SLICE_AXIS)
                if mask_slice.max() == 0:
                    continue
            elif is_mask and slice_2d.max() == 0:
                continue

        fname = f"{nii_path.stem}_slice_{i:04d}.png"
        save_slice(slice_2d, str(out_subdir / fname), RESIZE_TO)
        saved_count += 1

    return saved_count, n_slices


def main():
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Collect volume and mask files
    all_nii = sorted(input_dir.glob("*.nii"))
    volume_files = [f for f in all_nii if f.name.startswith("volume")]
    seg_files = [f for f in all_nii if f.name.startswith("segmentation")]

    total_saved = 0
    total_slices = 0

    # Convert masks first
    for seg_path in tqdm(seg_files, desc="Masks"):
        saved, total = process_nii_file(
            seg_path, masks_dir, is_mask=True
        )
        total_saved += saved
        total_slices += total

    # Convert CT volumes
    for vol_path in tqdm(volume_files, desc="Volumes"):
        idx = vol_path.stem.split("-")[-1]
        paired_mask = input_dir / f"segmentation-{idx}.nii"

        saved, total = process_nii_file(
            vol_path,
            images_dir,
            is_mask=False,
            mask_path=paired_mask if SKIP_EMPTY_MASK_SLICES else None
        )
        total_saved += saved
        total_slices += total

    print(f"\nSaved {total_saved} / {total_slices} slices")
    print(f"Output folder: {output_dir}")


if __name__ == "__main__":
    main()