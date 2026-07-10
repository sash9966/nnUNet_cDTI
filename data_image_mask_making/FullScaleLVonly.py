"""
Full-scale (NO crop) LV-only dataset for the Smart Health data.

Companion to FullScaleIPonly.py -- the two-model split of FullScaleLVandIP.py. Keeping LV
and the insertion points in ONE label map is wrong: the IP annotations are fixed-radius
DISKS that straddle the myocardial border, so merging them either carves the LV ring or
(via a union region) bulges it out into the RV. The 04 mask already stores LV / IP1 / IP2
as SEPARATE channels, so each model gets a clean target: here, the smooth LV ring (04 ch0).

Images: 03_Segmentation_Images (full-FoV avg / MD / eigenvector / FA), all 256x256.
Mask:   04_Segmentation_Masks ch0 (LV) -> label 1 (a complete, smooth ring).

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

pwd = '/Users/saschastocker/Documents/Stanford/DanEnnis20242025/Paper2025Automatic/Smart_Health'
root_folders = ['Hannum']
datasetname = 'Dataset121_HannumSmartHealthFullLV'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)


def normalize_image(image):
    image_min = np.min(image); image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

def normalise_MD(image):
    return (image - 0) / 4.0

def normalise_eigenvector(image):
    return image / np.sqrt(2)   # X+Y of a unit eigenvector -> capped at sqrt(2)


def lv_mask(mask3):
    """04 mask (256,256,3) [LV, IP1, IP2] -> smooth LV ring only, label 1."""
    out = np.zeros(mask3.shape[:2], dtype=np.uint8)
    out[mask3[:, :, 0] > 0] = 1
    return out


def save_inspection_plots(image_data, mask_data, filename_base):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(image_data, cmap='gray'); axes[0].set_title('Average Diffusion Image (full FoV)'); axes[0].axis('off')
    axes[1].imshow(mask_data, cmap='gray'); axes[1].set_title('LV mask (label 1)'); axes[1].axis('off')
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
                print(f'Missing quality information in {divo_folder}'); continue
            quality_data = pd.read_excel(excel_path)

            mask_folder = os.path.join(divo_path, '04_Segmentation_Masks')
            image_folder = os.path.join(divo_path, '03_Segmentation_Images')

            slice_files = glob.glob(os.path.join(mask_folder, 'Segmentation_Slice_*.nii'))
            slice_numbers = sorted(int(re.search(r'Slice_(\d+)\.nii$', f).group(1)) for f in slice_files)
            print(f'{volunteer_folder}/{divo_folder}: found {len(slice_numbers)} slices -> {slice_numbers}')

            for i in slice_numbers:
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
                    print(f'Failed to find required files for slice {i} in {divo_folder}'); continue

                mask_img = nib.load(mask_file)
                combined_mask = lv_mask(mask_img.get_fdata())

                avg_img = nib.load(avg_file)
                avg_image_data = normalize_image(avg_img.get_fdata())
                mean_diff_data = normalise_MD(nib.load(mean_diff_file).get_fdata())
                eigenvector_data = nib.load(eigenvector_file).get_fdata()
                combined_eigenvector_data = normalise_eigenvector(eigenvector_data[:, :, 0] + eigenvector_data[:, :, 1])
                FA_image_data = nib.load(FA_file).get_fdata()   # already [0,1]

                common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'
                for ch, arr in zip(range(4), [avg_image_data, mean_diff_data, combined_eigenvector_data, FA_image_data]):
                    nib.save(nib.Nifti1Image(arr, avg_img.affine),
                             os.path.join(output_image_folder, f'{common_name_id}_{ch:04d}.nii.gz'))
                nib.save(nib.Nifti1Image(combined_mask, avg_img.affine),
                         os.path.join(output_mask_folder, f'{common_name_id}.nii.gz'))
                save_inspection_plots(avg_image_data, combined_mask, common_name_id)
                print(f'Saved full-scale LV: {common_name_id}.nii.gz (LV {int((combined_mask==1).sum())} px)')
