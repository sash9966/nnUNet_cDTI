"""
Full-scale (NO crop) LV + insertion-point dataset for the Smart Health data.

Motivation: instead of the crop -> resample -> segment pipeline, train ONE nnUNet on the
FULL-FoV 4-contrast images that outputs LV (1) + both insertion points (2, 3) in a single
pass. If it works, a user just drops the 4 contrasts into Slicer, runs the nnUNet extension,
and gets the segmentation out -- no YOLO crop, no resample, no un-crop.

Data (per Volunteer/DiVO/slice, all already 256x256 and mutually aligned):
  03_Segmentation_Images/  -> full-scale modalities (avg, MD, primary eigenvector, FA)
  04_Segmentation_Masks/   -> full-scale GT mask, (256,256,3) = [LV, IP1, IP2]
So no un-cropping is needed: pair 03 with 04 and combine the 3 channels to 1/2/3.

Note: at full FoV the LV is ~1.7% of the image and each IP is ~0.1% (a few voxels) -- this
is the hard, class-imbalanced test. nnUNet's foreground oversampling is what makes it viable.

nnUNet is used with noNorm, so the per-channel normalization here IS the normalization:
  _0000 avg  -> per-image min-max            _0002 eigenvector (X+Y) -> /sqrt(2)
  _0001 MD   -> /4 (physical ceiling)        _0003 FA               -> raw [0,1]
"""
import os
import glob
import re
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Root directory and subfolders
pwd = '/Users/saschastocker/Documents/Stanford/DanEnnis20242025/Paper2025Automatic/Smart_Health'
root_folders = ['Hannum']
datasetname = 'Dataset120_HannumSmartHealthFullLVIP'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)


# ---- per-channel normalization (must match inference; nnUNet is noNorm) ----
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

def normalise_MD(image):
    image_min = 0
    image_max = 4
    return (image - image_min) / (image_max - image_min)

def normalise_eigenvector(image):
    # X and Y are components of a UNIT eigenvector, so their sum is capped at sqrt(2) (~1.414),
    # not 2. Divide by that fixed ceiling instead of min-max (stable; see the crop-based prep).
    return image / np.sqrt(2)


# ---- combine the full-scale 3-channel GT (LV, IP1, IP2) -> LV=1, IP1=2, IP2=3 ----
def combine_full_mask(mask3):
    out = np.zeros(mask3.shape[:2], dtype=np.uint8)
    out[mask3[:, :, 0] > 0] = 1   # LV myocardium
    out[mask3[:, :, 1] > 0] = 2   # anterior insertion point (IP1)
    out[mask3[:, :, 2] > 0] = 3   # inferior insertion point (IP2)
    return out


def save_inspection_plots(image_data, mask_data, filename_base):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_data, cmap='gray'); axes[0].set_title('Average Diffusion Image (full FoV)'); axes[0].axis('off')
    axes[1].imshow(mask_data, cmap='nipy_spectral'); axes[1].set_title('Mask (LV=1, IP1=2, IP2=3)'); axes[1].axis('off')
    axes[2].imshow(image_data, cmap='gray')
    axes[2].imshow(np.ma.masked_where(mask_data == 0, mask_data), cmap='autumn', alpha=0.6)
    axes[2].set_title('Overlay'); axes[2].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(inspection_folder, f'{filename_base}_inspection.png'))
    plt.close()


for root_folder in root_folders:
    print(f'root folder: {root_folder}')
    root_path = os.path.join(pwd, root_folder)

    for volunteer_folder in os.listdir(root_path):
        if not volunteer_folder.startswith('Volunteer'):
            continue
        distortion_corrected_folder = os.path.join(root_path, volunteer_folder, 'Distortion_Corrected')
        if not os.path.isdir(distortion_corrected_folder):
            continue

        for divo_folder in os.listdir(distortion_corrected_folder):
            if not (divo_folder.startswith('DiVO') or divo_folder.startswith('MDDW')):
                continue
            divo_path = os.path.join(distortion_corrected_folder, divo_folder)

            excel_path = os.path.join(divo_path, 'Detailed_Information.xlsx')
            if not os.path.exists(excel_path):
                print(f'Missing quality information in {divo_folder}')
                continue
            quality_data = pd.read_excel(excel_path)

            mask_folder = os.path.join(divo_path, '04_Segmentation_Masks')    # FULL-scale masks
            image_folder = os.path.join(divo_path, '03_Segmentation_Images')  # FULL-scale images

            # Discover slices dynamically from the full-scale mask folder.
            slice_files = glob.glob(os.path.join(mask_folder, 'Segmentation_Slice_*.nii'))
            slice_numbers = sorted(
                int(re.search(r'Slice_(\d+)\.nii$', f).group(1)) for f in slice_files
            )
            print(f'{volunteer_folder}/{divo_folder}: found {len(slice_numbers)} slices -> {slice_numbers}')

            for i in slice_numbers:
                # Per-slice quality gate (same as the crop-based prep).
                slice_data = quality_data[quality_data['Slice Number'] == i]
                if slice_data.empty or slice_data.iloc[0]['Image Quality'] != 'Good Image Quality':
                    print(f'Skipping slice {i} in {volunteer_folder}/{divo_folder} (bad quality)')
                    continue

                mask_file = os.path.join(mask_folder, f'Segmentation_Slice_{i:03d}.nii')
                avg_file = os.path.join(image_folder, f'Average_Diffusion_Weighted_Image_Slice_{i:03d}.nii')
                mean_diff_file = os.path.join(image_folder, f'Mean_Diffusivty_Image_Slice_{i:03d}.nii')
                eigenvector_file = os.path.join(image_folder, f'Primary_Eigenvector_Image_Slice_{i:03d}.nii')
                FA_file = os.path.join(image_folder, f'Fractional_Anisotropy_Image_Slice_{i:03d}.nii')

                if not all(os.path.exists(f) for f in [mask_file, avg_file, mean_diff_file, eigenvector_file, FA_file]):
                    print(f'Failed to find required files for slice {i} in {divo_folder}')
                    continue

                # GT: full-scale 3-channel mask -> single label mask (LV=1, IP1=2, IP2=3).
                mask_img = nib.load(mask_file)
                combined_mask = combine_full_mask(mask_img.get_fdata())

                # Images: full-scale modalities (no crop, no resize).
                avg_img = nib.load(avg_file)
                avg_image_data = avg_img.get_fdata()
                mean_diff_data = nib.load(mean_diff_file).get_fdata()
                eigenvector_data = nib.load(eigenvector_file).get_fdata()      # (256, 256, 3)
                FA_image_data = nib.load(FA_file).get_fdata()

                avg_image_data = normalize_image(avg_image_data)
                mean_diff_data = normalise_MD(mean_diff_data)
                combined_eigenvector_data = eigenvector_data[:, :, 0] + eigenvector_data[:, :, 1]
                combined_eigenvector_data = normalise_eigenvector(combined_eigenvector_data)
                # FA already in [0, 1]

                common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'

                # Save all four channels with the image affine, and the mask with the SAME affine
                # so nnUNet sees matching geometry (03 and 04 are co-registered full-FoV slices).
                nib.save(nib.Nifti1Image(avg_image_data, avg_img.affine),
                         os.path.join(output_image_folder, f'{common_name_id}_0000.nii.gz'))
                nib.save(nib.Nifti1Image(mean_diff_data, avg_img.affine),
                         os.path.join(output_image_folder, f'{common_name_id}_0001.nii.gz'))
                nib.save(nib.Nifti1Image(combined_eigenvector_data, avg_img.affine),
                         os.path.join(output_image_folder, f'{common_name_id}_0002.nii.gz'))
                nib.save(nib.Nifti1Image(FA_image_data, avg_img.affine),
                         os.path.join(output_image_folder, f'{common_name_id}_0003.nii.gz'))
                nib.save(nib.Nifti1Image(combined_mask, avg_img.affine),
                         os.path.join(output_mask_folder, f'{common_name_id}.nii.gz'))

                save_inspection_plots(avg_image_data, combined_mask, common_name_id)
                print(f'Saved full-scale LV+IP: {common_name_id}.nii.gz '
                      f'(LV {int((combined_mask==1).sum())} px, IP1 {int((combined_mask==2).sum())} px, '
                      f'IP2 {int((combined_mask==3).sum())} px)')
