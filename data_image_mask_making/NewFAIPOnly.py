


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
datasetname = 'Dataset105_HannumSmartHealthDataIPs'
output_mask_folder = f'{pwd}/{datasetname}/labelsTr'
output_image_folder = f'{pwd}/{datasetname}/imagesTr'
inspection_folder = f'{pwd}/inspection{datasetname}'

# Ensure output folders exist
os.makedirs(output_mask_folder, exist_ok=True)
os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(inspection_folder, exist_ok=True)

def process_mask_slices(mask_data,lv_only ):
    """
    Combines three 2D mask slices (256, 256 each) into a single mask with:
    - LV (slice 1): 1
    - Insertion point (slice 2): 2
    - Second insertion point (slice 3): 3
    Output is a single 2D mask with shape (256, 256).
    """
    # Initialize the combined mask with zeros (256, 256)
    combined_mask = np.zeros((256, 256), dtype=np.uint8)
    if mask_data.shape[0] == 256 and mask_data.shape[-1] == 3:
        # Reshape to (3, 256, 256)
        mask_data = np.transpose(mask_data, (2, 0, 1))

    # Add slice 1 (LV) with label 1
    
    if(lv_only):
        combined_mask[mask_data[0, :, :] == 1] = 1


    #IPs only!   
    else:
         # Add slice 2 (Insertion point) with label 2
        combined_mask[mask_data[1, :, :] == 1] = 1

        # Add slice 3 (Second insertion point) with label 3
        combined_mask[mask_data[2, :, :] == 1] = 2

    return combined_mask

# Function to normalize images to [0, 1] range
def normalize_image(image):
    image_min = np.min(image)
    image_max = np.max(image)
    return (image - image_min) / (image_max - image_min)

def normalise_MD(image):
    image_min = 0
    image_max = 4
    return (image - image_min) / (image_max - image_min)

# Function to save images for inspection
def save_inspection_plots(image_data, mask_data, filename_base):
    """Saves inspection plots of image, mask, and overlay using matplotlib."""
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Just the image (first channel)
    axes[0].imshow(image_data[:, :, 0], cmap='gray')  # Show first channel (average)
    axes[0].set_title('Average Diffusion Image')
    axes[0].axis('off')

    # Plot 2: Just the mask
    axes[1].imshow(mask_data, cmap='gray')
    axes[1].set_title('Mask Only')
    axes[1].axis('off')

    # Plot 3: Image with mask overlay
    axes[2].imshow(image_data[:, :, 0], cmap='gray')  # Show first channel (average)
    mask_overlay = np.ma.masked_where(mask_data != 0, mask_data)
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

                        # Check if the "Image Quality" column has "Good Image" for all rows (3 slices)
                        
                            # Proceed with the images only if all slices are of good quality

                    mask_slices = []
                    excel_path = os.path.join(divo_path, 'Detailed_Information.xlsx')
                    if os.path.exists(excel_path):
                        quality_data = pd.read_excel(excel_path)

                        mask_folder = os.path.join(divo_path, '06_Segmentation_Masks_CI')
                        image_folder = os.path.join(divo_path, '05_Segmentation_Images_CI')

                        # Count how many segmentation slices actually exist in this sub-folder
                        # (was hard-coded to the first 3 slices; folders can have many more, e.g. up to 12).
                        slice_files = glob.glob(os.path.join(mask_folder, 'Cropped_Segmentation_Slice_*.nii'))
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
                                mask_file = os.path.join(mask_folder, f'Cropped_Segmentation_Slice_{i:03d}.nii')
                                
                            
                                mask_img = nib.load(mask_file)

                                # Assuming the mask file is 3, 256, 256
                                mask_data = mask_img.get_fdata()  # Load the 3D mask (3, 256, 256)

                                # Process and combine the slices
                                #IPs only!!!
                                combined_mask = process_mask_slices(mask_data,lv_only=False)

                                
                                avg_image_file = os.path.join(image_folder, f'Cropped_Average_Diffusion_Weighted_Image_Slice_{i:03d}.nii')
                                mean_diff_file = os.path.join(image_folder, f'Cropped_Mean_Diffusivty_Image_Slice_{i:03d}.nii')
                                eigenvector_file = os.path.join(image_folder, f'Cropped_Primary_Eigenvector_Image_Slice_{i:03d}.nii')
                                FA_file = os.path.join(image_folder, f'Cropped_Fractional_Anisotropy_Image_Slice_{i:03d}.nii')

                                if os.path.exists(mask_file) and os.path.exists(avg_image_file) and os.path.exists(mean_diff_file) and os.path.exists(eigenvector_file):


                                    # Load the three NIfTI image files (average, mean diffusivity, eigenvector)
                                    avg_img = nib.load(avg_image_file)
                                    mean_diff_img = nib.load(mean_diff_file)
                                    eigenvector_img = nib.load(eigenvector_file)
                                    FA_image = nib.load(FA_file)

                                    # Get the data for all three images
                                    avg_image_data = avg_img.get_fdata()  # (256, 256)
                                    mean_diff_data = mean_diff_img.get_fdata()  # (256, 256)
                                    eigenvector_data = eigenvector_img.get_fdata()  # (256, 256, 3)
                                    FA_image_data = FA_image.get_fdata()

                                    # Normalize each image to [0, 1] range to prevent "washed out" effect
                                    avg_image_data = normalize_image(avg_image_data)
                                    mean_diff_data = normalise_MD(mean_diff_data)
                                    eigenvector_slice1 = eigenvector_data[:, :, 0]
                                    eigenvector_slice2 = eigenvector_data[:, :, 1]

                                    # Combine eigenvector slices 1 and 2 into a single slice
                                    combined_eigenvector_data = eigenvector_slice1 + eigenvector_slice2
                                    combined_eigenvector_data = normalize_image(combined_eigenvector_data)

                                    # Stack all three channels (average, mean diffusivity, combined eigenvector)
                                    combined_image_data = np.stack([avg_image_data, mean_diff_data, combined_eigenvector_data], axis=-1)

                                    # Save each channel separately (modality files with 0000, 0001, 0002 suffixes)
                                    common_name_id = f'{root_folder}_{volunteer_folder}_{divo_folder}_slice_{i:03d}'

                                    # Save Average Diffusion Image as _0000
                                    nib.save(nib.Nifti1Image(avg_image_data, avg_img.affine), 
                                            os.path.join(output_image_folder, f'{common_name_id}_0000.nii.gz'))

                                    # Save Mean Diffusivity Image as _0001
                                    nib.save(nib.Nifti1Image(mean_diff_data, mean_diff_img.affine), 
                                            os.path.join(output_image_folder, f'{common_name_id}_0001.nii.gz'))

                                    # Save Eigenvector Image (use one channel for now) as _0002
                                    nib.save(nib.Nifti1Image(combined_eigenvector_data, eigenvector_img.affine), 
                                            os.path.join(output_image_folder, f'{common_name_id}_0002.nii.gz'))
                                    
                                    nib.save(nib.Nifti1Image(FA_image_data, FA_image.affine), 
                                            os.path.join(output_image_folder, f'{common_name_id}_0003.nii.gz'))
                                    
                                

                                    # Save the mask
                                    nib.save(nib.Nifti1Image(combined_mask, mask_img.affine), os.path.join(output_mask_folder, f'{common_name_id}.nii.gz'))

                                    print(f'Saved mask slice {i}: {common_name_id}.nii.gz')
                                    print(f'Saved image modalities: {common_name_id}_0000.nii.gz, _0001.nii.gz, _0002.nii.gz')

                                    # Save inspection images (original, mask, overlay)
                                    save_inspection_plots(np.stack([avg_image_data, mean_diff_data, eigenvector_data[:, :, 1]], axis=-1), 
                                                        combined_mask, common_name_id)
                                else:
                                    print(f'Failed to find required files for slice {i}')
                        else:
                            # If any slice is not "Good Image", skip the processing and print the folder name
                            print(f'Skipping images, slice: {i} in volunter; {volunteer_folder} due to bad quality')

                    else:
                        print(f'Missing quality information in {divo_folder}')
