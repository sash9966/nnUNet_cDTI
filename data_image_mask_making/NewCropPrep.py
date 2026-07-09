import os
import glob
import re
import nibabel as nib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Root directory and subfolders
pwd = '/Users/saschastocker/Documents/Stanford/DanEnnis20242025/Paper2025Automatic/Smart_Health'
root_folders = ['Hannum']
datasetname = 'Dataset110_HannumSmarthHealthDataCrop'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'

# Ensure output folders existw
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)

# Function to normalize images to [0, 1] range
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

# Function to save images for inspection
def save_inspection_plots(image_data, mask_data, filename_base):
    """Saves inspection plots of image, mask, and overlay using matplotlib."""
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Just the image
    axes[0].imshow(image_data, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Plot 2: Just the mask
    axes[1].imshow(mask_data, cmap='gray')
    axes[1].set_title('Mask Only')
    axes[1].axis('off')

    # Plot 3: Image with mask overlay
    axes[2].imshow(image_data, cmap='gray')
    mask_overlay = np.ma.masked_where(mask_data != 1, mask_data)
    axes[2].imshow(mask_overlay, cmap='Reds', alpha=0.5)
    axes[2].set_title('Image with Mask Overlay')
    axes[2].axis('off')

    # Save the figure
    output_file = os.path.join(inspection_folder, f'{filename_base}_inspection.png')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close(fig)
    print(f'Saved inspection plot: {output_file}')


for root_folder in root_folders:
    print(f'root folder: {root_folder}')
    root_path = os.path.join(pwd, root_folder)

    # Loop through volunteer folders
    for volunteer_folder in os.listdir(root_path):
        if volunteer_folder.startswith('Volunteer'):
            volunteer_path = os.path.join(root_path, volunteer_folder)
            distortion_corrected_folder = os.path.join(volunteer_path, 'Distortion_Corrected')

            # Loop through DiVO and MDDW folders
            for divo_folder in os.listdir(distortion_corrected_folder):
                if divo_folder.startswith('DiVO') or divo_folder.startswith('MDDW'):
                    divo_path = os.path.join(distortion_corrected_folder, divo_folder)

                    # Load the quality control information from the Excel file in the DiVO/MDDW folder
                    excel_path = os.path.join(divo_path, 'Detailed_Information.xlsx')
                    if os.path.exists(excel_path):
                        quality_data = pd.read_excel(excel_path)

                        mask_folder = os.path.join(divo_path, '02_Crop_Masks')
                        image_folder = os.path.join(divo_path, '03_Segmentation_Images')

                        # Count how many crop-mask slices actually exist in this sub-folder
                        # (was hard-coded to the first 3 slices; folders can have many more, e.g. up to 12).
                        slice_files = glob.glob(os.path.join(mask_folder, 'Square_Crop_Mask_Slice_*.nii'))
                        slice_numbers = sorted(
                            int(re.search(r'Slice_(\d+)\.nii$', f).group(1)) for f in slice_files
                        )
                        num_slices = len(slice_numbers)
                        print(f'{volunteer_folder}/{divo_folder}: found {num_slices} slices -> {slice_numbers}')

                        # Loop through every slice that exists in this folder
                        for i in slice_numbers:
                            # Filter rows where "Slice Number" equals i
                            slice_data = quality_data[quality_data['Slice Number'] == i]
    
                            # Check if we found a matching row and if that row’s image quality is "Good Image Quality"
                            if not slice_data.empty and slice_data.iloc[0]['Image Quality'] == 'Good Image Quality':

                            
                                mask_folder = os.path.join(divo_path, '02_Crop_Masks')
                                image_folder = os.path.join(divo_path, '03_Segmentation_Images')
                                mask_folder = os.path.join(divo_path, '02_Crop_Masks')
                                image_folder = os.path.join(divo_path, '03_Segmentation_Images')
                                # Select mask and image files for each iteration
                                mask_file = os.path.join(mask_folder, f'Square_Crop_Mask_Slice_{i:03d}.nii')
                                image_file = os.path.join(image_folder, f'Average_Diffusion_Weighted_Image_Slice_{i:03d}.nii')

        
                                if os.path.exists(mask_file) and os.path.exists(image_file):
                                    # Load the NIfTI mask file and extract the 0th slice
                                    mask_img = nib.load(mask_file)
                                    mask_data = mask_img.get_fdata()

                                    # Load the NIfTI image file (no slicing needed)
                                    image_img = nib.load(image_file)
                                    image_data = image_img.get_fdata()
                                    image_data = normalize_image(image_data)

                                    common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'

                                    mask_output_filename = os.path.join(output_mask_folder,
                                                                        f'{common_name_id}.nii.gz') 
                                    image_output_filename = os.path.join(output_image_folder,
                                                                        f'{common_name_id}_0000.nii.gz')

                                    # Save the mask and image
                                    nib.save(nib.Nifti1Image(mask_data, mask_img.affine), mask_output_filename)
                                    nib.save(nib.Nifti1Image(image_data, image_img.affine), image_output_filename)

                                    print(f'Saved mask slice {i}: {mask_output_filename}')
                                    print(f'Saved image slice {i}: {image_output_filename}')

                                    # Save inspection images (original, mask, overlay)
                                
                                    save_inspection_plots(image_data, mask_data, common_name_id)
                                else: 
                                    if not os.path.exists(mask_file):
                                        print(f'Failed to find mask file: {mask_file}')
                                    if not os.path.exists(image_file):
                                        print(f'Failed to find image file: {image_file}')
                            else:
                                print(f' bad image quality, skipping')
                                print(f'Skipping images, slice: {i} in volunter; {volunteer_folder} due to bad quality')
                           
                    else:
                        print(f'Missing quality information in {divo_folder}')
