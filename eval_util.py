#WORKIGN 
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.ndimage import center_of_mass
from cardpy.Tools.Contours import Myocardial_Mask_Contour_Extraction

# Hausdorff distance function
def hausdorff_distance(a_points, b_points):

    #if no IP's are detected by nnUnet -> failsave, return distance 1000 
    if np.any(np.isnan(a_points)) or np.any(np.isinf(a_points)) or \
    np.any(np.isnan(b_points)) or np.any(np.isinf(b_points)):
        return 1000 
    if len(a_points) == 0:
        return 0 if len(b_points) == 0 else np.inf
    elif len(b_points) == 0:
        return np.inf
    return max(max(cKDTree(a_points).query(b_points, k=1)[0]),
               max(cKDTree(b_points).query(a_points, k=1)[0]))

# Average Hausdorff distance function
def average_hausdorff_distance(a_points, b_points):
    test_1 = cKDTree(a_points).query(b_points, k=1)[0]
    test_2 = cKDTree(b_points).query(a_points, k=1)[0]
    AHD = ((1/len(a_points) * np.sum(test_1)) + (1/len(b_points) * np.sum(test_2))) / 2
    return AHD

# Calculate center of mass (centroid) of a mask label
def calculate_center_of_mass(mask, label_value):
    return center_of_mass(mask == label_value)

# Load NIfTI files for the masks and original images, including the affine matrix
def load_nifti_with_affine(file_path):
    nifti_file = nib.load(file_path)
    return nifti_file.get_fdata(), nifti_file.affine

# Convert pixel distances to real-world distances using voxel spacing
def convert_to_physical_distance(pixel_distance, spacing):
    return pixel_distance * spacing

def calculate_all_points(mask, label_value):
    points = np.argwhere(mask == label_value)
    return points 



def calculate_scale_factor(original_affine, crop_mask_path, target_size=256):
    """
    Load the crop mask, force a square bounding box, and then compute
    a single scale factor = (side_length / target_size).
    """

    #spacing is 1x1 mm so we can ignore the affine matrix to calculate the spacing 
    # 1. Load crop mask
    crop_mask, _ = load_nifti_with_affine(crop_mask_path)

    # 2. Identify bounding box for all non-zero pixels
    coords = np.argwhere(crop_mask > 0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # +1 so x_max, y_max are exclusive

    # 3. Force a square
    side_length = max(x_max - x_min, y_max - y_min)

    # Re-center so bounding box is square around the same center
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    x_min = max(0, center_x - side_length // 2)
    x_max = x_min + side_length
    y_min = max(0, center_y - side_length // 2)
    y_max = y_min + side_length

    # 4. Derive the scale factor
    # Since it's square, we only need one scale factor
    scale_factor = side_length / target_size

    return scale_factor


def precision_score_(groundtruth_mask, pred_mask):
    """
    Calculate precision score for a binary mask comparison.
    
    Parameters:
    - groundtruth_mask: Ground truth binary mask (numpy array)
    - pred_mask: Predicted binary mask (numpy array)
    
    Returns:
    - precision: Precision score rounded to 3 decimal places
    """
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_pred = np.sum(pred_mask)
    if total_pixel_pred == 0:  # Handle division by zero
        precision = 0.0 if np.sum(groundtruth_mask) > 0 else 1.0
    else:
        precision = intersect / total_pixel_pred
    return round(precision, 3)

def recall_score_(groundtruth_mask, pred_mask):
    """
    Calculate recall score for a binary mask comparison.
    
    Parameters:
    - groundtruth_mask: Ground truth binary mask (numpy array)
    - pred_mask: Predicted binary mask (numpy array)
    
    Returns:
    - recall: Recall score rounded to 3 decimal places
    """
    intersect = np.sum(pred_mask * groundtruth_mask)
    total_pixel_truth = np.sum(groundtruth_mask)
    if total_pixel_truth == 0:  # Handle division by zero
        recall = 1.0 if np.sum(pred_mask) == 0 else 0.0
    else:
        recall = intersect / total_pixel_truth
    return round(recall, 3)

def segmentation_f1_score(gt_mask, pred_mask, label=1):
    """
    Calculate the F1 Score for a specific label using provided precision and recall definitions.
    F1 Score is the harmonic mean of precision and recall, providing a balanced measure
    of agreement between ground truth and prediction.

    Parameters:
    - gt_mask: Ground truth mask (numpy array)
    - pred_mask: Predicted mask (numpy array)
    - label: The label value to evaluate (default=1)

    Returns:
    - f1: F1 Score, rounded to 3 decimal places
    """
    # Binarize masks for the specific label
    gt = (gt_mask == label).astype(np.float32)
    pred = (pred_mask == label).astype(np.float32)

    # Calculate precision and recall using provided definitions
    precision = precision_score_(gt, pred)
    recall = recall_score_(gt, pred)

    # Calculate F1 score as the harmonic mean of precision and recall
    if precision == 0 and recall == 0:
        f1 = 0.0  # Avoid division by zero; F1 is 0 if both precision and recall are 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    return round(f1, 3), precision, recall

def dice_score_original(gt_mask, pred_mask, label=1):
    """
    Calculate the Dice score on the GT and prediction masks for label 1 (left ventricle).
    """
    gt = (gt_mask == label)
    pred = (pred_mask == label)
    intersection = np.logical_and(gt, pred).sum()
    gt_sum = gt.sum()
    pred_sum = pred.sum()
    dice = (2. * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 1
    return dice

def histogram_comparison(gt_mask, pred_mask, original_image, label=1):
    """
    Compare histograms of label 1 from gt_mask and pred_mask based on the intensities
    in the segmented area in the original_image. Plots a comparison view and calculates
    a mean percentage difference.
    """
    gt_mask_label = (gt_mask == label)
    pred_mask_label = (pred_mask == label)
    
    gt_intensities = original_image[gt_mask_label]
    pred_intensities = original_image[pred_mask_label]
    
    # Plot histograms with custom colors
   
    # Calculate median intensities

    #ADD TO OUTPUT FOR EXCEL, NEEDED FOR BLANT ALTMAN COMPARISON
    gt_median_intensity = np.median(gt_intensities)
    pred_median_intensity = np.median(pred_intensities)

    # Create histogram plot
    plt.figure()
    plt.hist(gt_intensities.ravel(), bins=50, alpha=0.5, color='blue', label='Ground Truth')
    plt.hist(pred_intensities.ravel(), bins=50, alpha=0.5, color='red', label='Prediction')

    # Highlight the median values
    plt.axvline(gt_median_intensity, color='blue', linestyle='dashed', linewidth=1.5, label=f'GT Median: {gt_median_intensity:.2f}')
    plt.axvline(pred_median_intensity, color='red', linestyle='dashed', linewidth=1.5, label=f'Pred Median: {pred_median_intensity:.2f}')

    # Add annotations for median values
    plt.text(gt_median_intensity, plt.ylim()[1] * 0.9, f'{gt_median_intensity:.2f}', color='blue', ha='center')
    plt.text(pred_median_intensity, plt.ylim()[1] * 0.9, f'{pred_median_intensity:.2f}', color='red', ha='center')

    # Set legend, title, labels
    plt.legend()
    plt.title(f'Intensity Histogram Comparison for Label {label}')
    plt.xlabel('Intensity')
    plt.ylabel('Frequency')

    # Show plot
    plt.show()

    
    
    #Mean percentage difference
    if gt_median_intensity != 0:
        median_percentage_difference = abs(gt_median_intensity - pred_median_intensity) / gt_median_intensity % 100
    else:
        median_percentage_difference = 0.0  # Avoid division by zero

        #Compute histograms with fixed bins
    bins = 50  # Adjust the number of bins based on your image intensity range
    range_min = np.min(original_image)
    range_max = np.max(original_image)
    counts_gt, bin_edges = np.histogram(gt_intensities, bins=bins, range=(range_min, range_max))
    counts_pred, _ = np.histogram(pred_intensities, bins=bins, range=(range_min, range_max))

    # Calculate percentage differences for each intensity bin
    # percentage_differences = []
    # for gt_count, pred_count in zip(counts_gt, counts_pred):
    #     if gt_count == 0 and pred_count == 0:
    #         percentage_difference = 0.0
    #     else:
    #         # Use symmetric mean absolute percentage error to avoid division by zero issues
    #         denominator = (abs(gt_count) + abs(pred_count)) / 2
    #         percentage_difference = abs(gt_count - pred_count) / denominator * 100
    #     percentage_differences.append(percentage_difference)

    # Calculate the mean of the percentage differences
    #mean_percentage_difference_individual = np.mean(percentage_differences)

    

    ####TOODOO!!
    return gt_median_intensity, pred_median_intensity,median_percentage_difference, 


def evaluate_hausdorff_downsampled(pred_file, gt_file, scale_factor, original_image, filename):
    # Load the downsampled prediction and ground truth masks
    print(f'pred_file : {pred_file}')
    pred_mask, _ = load_nifti_with_affine(pred_file)

    gt_mask, _ = load_nifti_with_affine(gt_file)
    
    binary_pred_mask = (pred_mask >0).astype(np.uint8)
    binary_gt_mask   = (gt_mask >0).astype(np.uint8)
    if binary_pred_mask.ndim == 2:
        binary_pred_mask = binary_pred_mask[:, :, np.newaxis]
    if binary_gt_mask.ndim == 2:
        binary_gt_mask = binary_gt_mask[:, :, np.newaxis]

 





    
    # Calculate center points for downsampled masks
    pred_center_2 = calculate_center_of_mass(pred_mask, 2)
    gt_center_2 = calculate_center_of_mass(gt_mask, 2)
    pred_center_3 = calculate_center_of_mass(pred_mask, 3)
    gt_center_3 = calculate_center_of_mass(gt_mask, 3)
    
    # Hausdorff distance in pixels for downsampled mask
    hausdorff_2_pixels = hausdorff_distance([pred_center_2], [gt_center_2])
    hausdorff_3_pixels = hausdorff_distance([pred_center_3], [gt_center_3])
    
    # Convert pixel distances to physical distances using scale_factor
    hausdorff_2_mm = hausdorff_2_pixels * scale_factor
    hausdorff_3_mm = hausdorff_3_pixels * scale_factor
    
    
    # **Ensure the original image matches the mask dimensions**
    if original_image.shape != pred_mask.shape:
        from skimage.transform import resize
        original_image_resized = resize(original_image, pred_mask.shape, preserve_range=True, anti_aliasing=True)
    else:
        original_image_resized = original_image
    
    # Plot downsampled masks with original image for visual validation
    plot_masks_and_hausdorff(original_image_resized, pred_mask, gt_mask, pred_center_2, gt_center_2, pred_center_3, gt_center_3, filename)
    
    return {
        
        "Hausdorff_2 (pixels)": hausdorff_2_pixels,
        "Hausdorff_2 (mm)": hausdorff_2_mm,
        "Hausdorff_3 (pixels)": hausdorff_3_pixels,
        "Hausdorff_3 (mm)": hausdorff_3_mm,
    }


def evaluate_hausdorff_original(original_pred_mask, original_gt_mask, original_image, filename):


    binary_pred_mask = (original_pred_mask >0).astype(np.uint8)
    binary_gt_mask   = (original_gt_mask >0).astype(np.uint8)
    if binary_pred_mask.ndim == 2:
        binary_pred_mask = binary_pred_mask[:, :, np.newaxis]
    if binary_gt_mask.ndim == 2:
        binary_gt_mask = binary_gt_mask[:, :, np.newaxis]

    # Calculate center points for original masks
    pred_center_2 = calculate_center_of_mass(original_pred_mask, 2)
    gt_center_2 = calculate_center_of_mass(original_gt_mask, 2)
    pred_center_3 = calculate_center_of_mass(original_pred_mask, 3)
    gt_center_3 = calculate_center_of_mass(original_gt_mask, 3)
    
    # Hausdorff distance in pixels for original mask (pixels correspond to mm)
    hausdorff_2_pixels = hausdorff_distance([pred_center_2], [gt_center_2])
    hausdorff_3_pixels = hausdorff_distance([pred_center_3], [gt_center_3])
    


    epicardium_contours_pred, endocardium_contours_pred, endocardial_centers_pred = Myocardial_Mask_Contour_Extraction(binary_pred_mask,0)
    epicardium_contours_gt, endocardium_contours_gt, endocardial_centers_gt = Myocardial_Mask_Contour_Extraction(binary_gt_mask,0)
    # Transpose and assign back to variables
    #Shape to 256x256x1
    epicardium_contours_gt = np.transpose(epicardium_contours_gt, (1, 2, 0))
    epicardium_contours_pred = np.transpose(epicardium_contours_pred, (1, 2, 0))
    endocardium_contours_pred = np.transpose(endocardium_contours_pred, (1, 2, 0))
    endocardium_contours_gt = np.transpose(endocardium_contours_gt, (1, 2, 0))
    # Print shapes after transposes

    # # Plot

    # plt.figure(figsize=(15, 10))

    # # First row: Ground Truth Masks (Binary, Epicardium, Endocardium)
    # plt.subplot(2, 4, 1)
    # plt.imshow(binary_gt_mask, cmap='gray')
    # plt.title("Binary GT Mask")

    # plt.subplot(2, 4, 2)
    # plt.imshow(epicardium_contours_gt[:, :, 0], cmap='gray')
    # plt.title("Epicardium GT")

    # plt.subplot(2, 4, 3)
    # plt.imshow(endocardium_contours_gt[:, :, 0], cmap='gray')
    # plt.title("Endocardium GT")

    # # Second row: Prediction Masks (Binary, Epicardium, Endocardium)
    # plt.subplot(2, 4, 5)
    # plt.imshow(binary_pred_mask, cmap='gray')
    # plt.title("Binary Pred Mask")

    # plt.subplot(2, 4, 6)
    # plt.imshow(epicardium_contours_pred[:, :, 0], cmap='gray')
    # plt.title("Epicardium Pred")

    # plt.subplot(2, 4, 7)
    # plt.imshow(endocardium_contours_pred[:, :, 0], cmap='gray')
    # plt.title("Endocardium Pred")

    # # Adjust layout and display
    # plt.tight_layout()
    # plt.show()
    avg_hausdorff_epi = average_hausdorff_distance(epicardium_contours_pred[:,:,0], epicardium_contours_gt[:,:,0])
    avg_hausdorff_endo = average_hausdorff_distance(endocardium_contours_pred[:,:,0], endocardium_contours_gt[:,:,0])
    #avg_hausdorff_endocardial = average_hausdorff_distance(endocardial_centers_pred, endocardial_centers_gt)


    # Plot original image with original masks and Hausdorff distance for validation
    plot_original_with_masks(original_image, original_pred_mask, original_gt_mask, scale_factor=1.0, filename=filename)
    
    return {
        "Hausdorff_2 (pixels/mm)": hausdorff_2_pixels,
        "Hausdorff_3 (pixels/mm)": hausdorff_3_pixels,
        "Avg. HD endo": avg_hausdorff_endo,
        "Avg. HD epi": avg_hausdorff_epi,
    }


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd

def plot_original_with_masks(original_image, upscaled_pred_mask, upscaled_gt_mask, scale_factor, filename):
    """
    Plots the original image with overlaid upscaled prediction and ground truth masks, 
    and visualizes the Hausdorff distance between the mask center points.
    """
    # Calculate center points for labels 2 and 3
    pred_center_2 = calculate_center_of_mass(upscaled_pred_mask, 2)
    gt_center_2 = calculate_center_of_mass(upscaled_gt_mask, 2)
    pred_center_3 = calculate_center_of_mass(upscaled_pred_mask, 3)
    gt_center_3 = calculate_center_of_mass(upscaled_gt_mask, 3)

    # Calculate Hausdorff distances for label 2 and label 3
    hausdorff_2 = hausdorff_distance([pred_center_2], [gt_center_2])
    hausdorff_3 = hausdorff_distance([pred_center_3], [gt_center_3])

    # Convert pixel distances to real-world distances using the scale factor
    hausdorff_2_physical = hausdorff_2 * scale_factor
    hausdorff_3_physical = hausdorff_3 * scale_factor

    # # Define custom colormaps for prediction and ground truth masks
    # Prediction mask colormap: [Background, LV, Label 2, Label 3]
    pred_colors = [
        (0, 0, 0, 0),          # 0 - Transparent
         (0, 1, 1, 0.5),       # 1 - Cyan   
        (0, 0, 1, 0.5),        # 2 - Blue (Label 2)
        (0, 0, 1, 0.5),        # 3 - Blue (Label 3)
    ]
    pred_cmap = ListedColormap(pred_colors)
    
    # Ground truth mask colormap: [Background, LV, Label 2, Label 3]
    gt_colors = [
        (0, 0, 0, 0),          # 0 - Transparent
        (1, 1, 0, 0.5),         # 1 - Yellow
        (1, 0, 0, 0.5),        # 2 - Red (Label 2)
        (1, 0, 0, 0.5)         # 3 - Red (Label 3)
    ]
    gt_cmap = ListedColormap(gt_colors)

    # Plot original image
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap='gray')

    # Overlay ground truth mask
    plt.imshow(upscaled_gt_mask, cmap=gt_cmap, interpolation='none')

    # Overlay prediction mask
    plt.imshow(upscaled_pred_mask, cmap=pred_cmap, interpolation='none')

    # Plot center points and Hausdorff lines for label 2
    plt.scatter(*pred_center_2[::-1], color='blue', marker='x', s=100, label='Pred center of mass ')
    plt.scatter(*gt_center_2[::-1], color='red', marker='+', s=100, label='GT center of mass ')
    plt.plot([pred_center_2[1], gt_center_2[1]], [pred_center_2[0], gt_center_2[0]], 'b--', 
             label=f'HD superior IP')

    # Plot center points and Hausdorff lines for label 3
    #plt.scatter(*pred_center_3[::-1], color='blue', marker='x', s=100, label='Prediction inferior 3')
    #plt.scatter(*gt_center_3[::-1], color='red', marker='+', s=100, label='GT inferior ')
    plt.plot([pred_center_3[1], gt_center_3[1]], [pred_center_3[0], gt_center_3[0]], 'm--', 
             label=f'HD inferior IP')

    # Create legend handles for the GT masks
    gt_patches = [
        Patch(facecolor=gt_colors[1], edgecolor='none', label='GT LV Mask '),
        #Patch(facecolor=gt_colors[3], edgecolor='none', label='GT Inferior ')
    ]

    # Create legend handles for the Prediction masks
    pred_patches = [
        Patch(facecolor=pred_colors[1], edgecolor='none', label='Pred LV Mask '),
        Patch(facecolor=gt_colors[2], edgecolor='none', label='GT IP'),
        Patch(facecolor=pred_colors[2], edgecolor='none', label='Pred IP '),
       #Patch(facecolor=pred_colors[3], edgecolor='none', label='Pred Inferior ')
    ]

    # Get existing handles and labels from the plot
    handles, labels = plt.gca().get_legend_handles_labels()

    # Combine all handles
    all_handles = gt_patches + pred_patches + handles

    # Display legend and remove axes
    #plt.legend(handles=all_handles, loc='upper right', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    plt.axis('off')
    plt.show()

    fig_legend = plt.figure(figsize=(10, 6), dpi=300)
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')

    # Add the legend to the new figure
    legend = ax_legend.legend(handles=all_handles, loc='center', frameon=False)

    # Adjust layout and show the legend figure
    plt.tight_layout()
    plt.show()

def evaluate_hausdorff_and_plot(pred_mask, gt_mask, original_image, scale_factor, filename):
    # Calculate center points for labels in upscaled masks
    pred_center_2 = calculate_center_of_mass(pred_mask, 2)
    gt_center_2 = calculate_center_of_mass(gt_mask, 2)
    pred_center_3 = calculate_center_of_mass(pred_mask, 3)
    gt_center_3 = calculate_center_of_mass(gt_mask, 3)

    # Calculate Hausdorff distances in pixel units
    hausdorff_2 = hausdorff_distance([pred_center_2], [gt_center_2])
    avg_hausdorff_2 = average_hausdorff_distance([pred_center_2], [gt_center_2])
    hausdorff_3 = hausdorff_distance([pred_center_3], [gt_center_3])
    avg_hausdorff_3 = average_hausdorff_distance([pred_center_3], [gt_center_3])

    # Convert to real-world distances using the scale factor
    hausdorff_2_physical = convert_to_physical_distance(hausdorff_2, scale_factor)
    avg_hausdorff_2_physical = convert_to_physical_distance(avg_hausdorff_2, scale_factor)
    hausdorff_3_physical = convert_to_physical_distance(hausdorff_3, scale_factor)
    avg_hausdorff_3_physical = convert_to_physical_distance(avg_hausdorff_3, scale_factor)

    # Print out Hausdorff results to confirm calculations
    print(f"Filename: {filename}")
    print(f"Hausdorff 2 (pixels): {hausdorff_2}, (mm): {hausdorff_2_physical}")
    print(f"Avg Hausdorff 2 (pixels): {avg_hausdorff_2}, (mm): {avg_hausdorff_2_physical}")
    print(f"Hausdorff 3 (pixels): {hausdorff_3}, (mm): {hausdorff_3_physical}")
    print(f"Avg Hausdorff 3 (pixels): {avg_hausdorff_3}, (mm): {avg_hausdorff_3_physical}")

    # Plot everything for visual inspection

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_masks_and_hausdorff(original_image, pred_mask, gt_mask, pred_center_2, gt_center_2, pred_center_3, gt_center_3, filename):
    #plt.figure(figsize=(8, 8))
    #plt.title(f"Overlay of Downsampled Original Image and Masks: {filename}")
    
    # Display the original image
    plt.imshow(original_image, cmap='gray')
    plt.show()
    plt.imshow(original_image, cmap='gray')
    
    # Define custom colormaps for prediction and ground truth masks
    # Prediction mask colormap: [Background, LV, Label 2, Label 3]
    pred_colors = [
        (0, 0, 0, 0),          # 0 - Transparent
         (0, 1, 1, 0.5),       # 1 - Cyan   
        (0, 0, 1, 0.5),        # 2 - Blue (Label 2)
        (0, 0, 1, 0.5),        # 3 - Blue (Label 3)
    ]
    pred_cmap = ListedColormap(pred_colors)
    
    # Ground truth mask colormap: [Background, LV, Label 2, Label 3]
    gt_colors = [
        (0, 0, 0, 0),          # 0 - Transparent
        (1, 1, 0, 0.5),         # 1 - Yellow
        (1, 0, 0, 0.5),        # 2 - Red (Label 2)
        (1, 0, 0, 0.5)         # 3 - Red (Label 3)
    ]
    gt_cmap = ListedColormap(gt_colors)
    
    # # Overlay ground truth mask
    plt.imshow(gt_mask, cmap=gt_cmap, interpolation='none',label = 'GT LV Mask' )
    
    # Overlay prediction mask
    plt.imshow(pred_mask, cmap=pred_cmap, interpolation='none',label = 'Pred LV Mask')
    
    # Plot center points and Hausdorff lines for label 2
    plt.scatter(*pred_center_2[::-1], color='blue', label='Prediction Center 2', marker='x', s=100)
    plt.scatter(*gt_center_2[::-1], color='red', label='Ground Truth Center 2', marker='+', s=100)
    plt.plot([pred_center_2[1], gt_center_2[1]], [pred_center_2[0], gt_center_2[0]], 'b--', label='Hausdorff Distance 2')
    
    # Plot center points and Hausdorff lines for label 3
    plt.scatter(*pred_center_3[::-1], color='blue', label='Prediction Center 3', marker='x', s=100)
    plt.scatter(*gt_center_3[::-1], color='red', label='Ground Truth Center 3', marker='+', s=100)
    plt.plot([pred_center_3[1], gt_center_3[1]], [pred_center_3[0], gt_center_3[0]], 'm--', label='Hausdorff Distance 3')
    
    #plt.legend(loc='upper right')
    plt.axis('off')
    plt.show()



from scipy.ndimage import zoom

def upscale_segmentation_to_original(segmentation, crop_mask_path, target_size=256):
    """
    Upscale a 256x256 segmentation back into the original image space using
    the same approach as make_square_crop(). Specifically:
      - Find a square bounding box in the crop mask.
      - Compute a single scale factor = side_length / target_size.
      - Upscale with nearest-neighbor interpolation.
      - Place the upscaled segmentation back at the correct position
        in the full-size mask.
    """
    from scipy.ndimage import zoom
    
    # Load the crop mask to determine the original crop dimensions
    crop_mask, _ = load_nifti_with_affine(crop_mask_path)
    
    # Identify the bounding box of the region where the mask equals 1
    coords = np.argwhere(crop_mask>0)
    x_min, y_min = coords.min(axis=0)
    x_max, y_max = coords.max(axis=0) + 1  # +1 so x_max/y_max are exclusive
    
    # side_length = size of the square
    side_length = max(x_max - x_min, y_max - y_min)
    
    # Adjust bounds to make sure we have a square
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    
    x_min = max(0, center_x - side_length // 2)
    x_max = x_min + side_length
    y_min = max(0, center_y - side_length // 2)
    y_max = y_min + side_length
    
    # Compute scale factor for the square bounding box
    scale_factor = side_length / target_size
    
    # Nearest-neighbor interpolation preserves label values
    upscaled_segmentation = zoom(segmentation, (scale_factor, scale_factor), order=0)
    
    # Create an empty mask with the original image size
    full_size_mask = np.zeros_like(crop_mask)
    
    # Place the upscaled segmentation into the correct bounding box
    full_size_mask[x_min : x_min + upscaled_segmentation.shape[0], 
                   y_min : y_min + upscaled_segmentation.shape[1]] = upscaled_segmentation
    
    print("[UPSCALE] Computed bounding box from 'crop_mask_path':", crop_mask_path)
    print(f"  x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}, side_length={side_length}, scale_factor={scale_factor}")

    
    return full_size_mask





def process_folders(pred_folder, gt_folder, main_folder, metrics_data, metrics_entry_median, interreaderName,annotator,mask_inference_folder):
    metrics_data= metrics_data
    metrics_entry_median= metrics_entry_median


    for pred_file in os.listdir(pred_folder):
        if pred_file.endswith('.nii.gz'):
            pred_path = os.path.join(pred_folder, pred_file)
            gt_path = os.path.join(gt_folder, pred_file)
        if (interreaderName is not None):
            parts = pred_file.split('_')
            temp = pred_file
            gt_file = temp.replace(parts[0],interreaderName)
            gt_path = os.path.join(gt_folder, gt_file)
        
        print(f'gt_path: {gt_path}, pred_path: {pred_path}')
        if os.path.exists(gt_path):
            # For later, csv file generation/join with nnUnet output
            case_id = pred_file.replace('.nii.gz','')
                # e.g., '002'

            parts = pred_file.split('_')
            volunteer_id = f"{parts[1]}_{parts[2]}"
            divo_folder = f"{parts[3]}_{parts[4]}_{parts[5]}"
            slice_name = parts[-1].replace('.nii.gz', '.nii')  # Adjust for .nii extension
            slice_number = parts[-1].replace('.nii.gz', '') 
            # Construct paths to the original image slice and crop mask



            ###CHANGE TO mean diffusivity value! FOR MEAN INSTENISTY VALUE
            original_image_path_MD = os.path.join(
                main_folder,
                parts[0], volunteer_id, 'Distortion_Corrected', divo_folder, 
                '03_Segmentation_Images', 'Mean_Diffusivty_Image_Slice_' + slice_name
            )
            original_image_path_FA = os.path.join(
                main_folder,
                parts[0], volunteer_id, 'Distortion_Corrected', divo_folder, 
                '03_Segmentation_Images', 'Fractional_Anisotropy_Image_Slice_' + slice_name
            )

            print(f'original_image_path: {original_image_path_MD}')
            crop_mask_filename = f"{annotator}_{volunteer_id}_{divo_folder}_slice_{slice_number}.nii.gz"
        
            crop_mask_path = os.path.join(mask_inference_folder, crop_mask_filename)
            if (interreaderName is not None):
                divo_path = f"{main_folder}/{annotator}/{volunteer_id}/Distortion_Corrected/{divo_folder}"
                mask_folder = os.path.join(divo_path, "02_Crop_Masks")
                gt_mask_path = os.path.join(mask_folder, f"Square_Crop_Mask_Slice_{slice_number}.nii")  # or .nii.gz?
                crop_mask_path = gt_mask_path
            downsampled_original_image_path_AVG = os.path.join(
                main_folder,
                parts[0], volunteer_id, 'Distortion_Corrected', divo_folder,
                '05_Segmentation_Images_CI', 'Cropped_Average_Diffusion_Weighted_Image_Slice_' + slice_name
            )
            #print(f'original image_path = {original_image_path}, crop mask path: {crop_mask_path}')
            if os.path.exists(original_image_path_MD) and os.path.exists(crop_mask_path):
                # Load the original image
                original_image_MD, original_affine_MD = load_nifti_with_affine(original_image_path_MD)
                original_image_FA, original_affine_FA = load_nifti_with_affine(original_image_path_FA)
                
                # Calculate scale factor
                scale_factor  = calculate_scale_factor(original_affine_MD, crop_mask_path)
                
                # Load downsampled prediction and ground truth masks
                # These are already at 256x256
                pred_mask, _ = load_nifti_with_affine(pred_path)
                gt_mask, _ = load_nifti_with_affine(gt_path)
                downsampled_original_image_AVG, _ = load_nifti_with_affine(downsampled_original_image_path_AVG)


                # Evaluate downsampled masks
                #No longer needed, only intersted in realphysical world coordinates, need mm! Just for plots!
                results_downsampled = evaluate_hausdorff_downsampled(pred_path, gt_path, scale_factor, downsampled_original_image_AVG,pred_file)

                
                # Upscale masks to original dimensions.
                # The predicted LV mask was produced in the *predicted* crop frame,
                # so it is correctly un-cropped with the predicted crop box (crop_mask_path).
                original_pred_mask = upscale_segmentation_to_original(pred_mask, crop_mask_path)

                # BUGFIX (crop misalignment): the GT LV mask was annotated in the
                # *ground-truth* crop frame. Un-cropping it with the predicted crop box
                # re-projects it through the wrong square, shifting GT relative to pred by
                # the predicted-vs-GT crop offset and underreporting Dice. Use the GT square
                # crop box instead. In interreader mode crop_mask_path is ALREADY the GT crop
                # (both inputs are expert masks in the GT frame), so keep it there.
                if interreaderName is not None:
                    gt_crop_mask_path = crop_mask_path
                else:
                    gt_crop_mask_path = os.path.join(
                        main_folder, parts[0], volunteer_id, 'Distortion_Corrected', divo_folder,
                        '02_Crop_Masks', f'Square_Crop_Mask_Slice_{slice_number}.nii'
                    )
                original_gt_mask = upscale_segmentation_to_original(gt_mask, gt_crop_mask_path)

                # Save original masks (optional)
                original_pred_path = os.path.join(main_folder, parts[0], volunteer_id, 'Distortion_Corrected', divo_folder,
                                                    '01_Original_Images', f'upsampled_pred_Slice{slice_number}.nii.gz')
                original_gt_path = os.path.join(main_folder, parts[0], volunteer_id, 'Distortion_Corrected', divo_folder,
                                                '01_Original_Images', f'upsampled_gt_Slice_{slice_number}.nii.gz')
                

                nib.save(nib.Nifti1Image(original_pred_mask, original_affine_MD), original_pred_path)
                nib.save(nib.Nifti1Image(original_gt_mask, original_affine_MD), original_gt_path)
                
                # Evaluate original masks
                results_original = evaluate_hausdorff_original(original_pred_mask, original_gt_mask, original_image_MD, pred_file)
                hausdorff_distance_label2 = results_original.get('Hausdorff_2 (pixels/mm)', np.nan)
                hausdorff_distance_label3 = results_original.get('Hausdorff_3 (pixels/mm)', np.nan)
                avg_hausdorff_epi= results_original.get('Avg. HD epi',np.nan)
                avg_hausdorff_endo= results_original.get('Avg. HD endo',np.nan)
                

                
                
                dice_original_1= dice_score_original(original_gt_mask, original_pred_mask, label=1)
                #dice_original_2= dice_score_original(original_gt_mask, original_pred_mask, label=2)
                #dice_original_3= dice_score_original(original_gt_mask, original_pred_mask, label=3)
                f1_original, precision, recall = segmentation_f1_score(original_gt_mask, original_pred_mask, label=1)

                # Compare histograms of intensities in the segmented areas
                gt_median_MD, pred_median_MD ,median_percentage_difference_MD  = histogram_comparison(original_gt_mask, original_pred_mask, original_image_MD, label=1)

                gt_median_FA, pred_median_FA ,median_percentage_difference_FA  = histogram_comparison(original_gt_mask, original_pred_mask, original_image_FA, label=1)

                #Readjust to 1.25mm for right [mm]!
                hausdorff_distance_label2=1.25*hausdorff_distance_label2
                hausdorff_distance_label3=1.25*hausdorff_distance_label3
                avg_hausdorff_epi=1.25*avg_hausdorff_epi
                avg_hausdorff_endo=1.25*avg_hausdorff_endo
                metric_entry_median= {
                                        
                }
                
                metrics_entry = {
                    'Case ID': case_id,
                    'Slice Number': slice_number,
                    'Dice Score Original Label 1': dice_original_1,
                    #'Dice Score Original Label 2': dice_original_2,
                    #'Dice Score Original Label 3': dice_original_3,
                    'Hausdorff Distance Label 2': hausdorff_distance_label2,
                    'Hausdorff Distance Label 3': hausdorff_distance_label3,
                    'GT_Median_MD': gt_median_MD,
                    'Pred_median_MD':pred_median_MD,
                    'Mean Percentage Difference Label 1 GT MD': median_percentage_difference_MD,
                    'GT_Median_FA': gt_median_FA,
                    'Pred_median_FA':pred_median_FA,
                    'Mean Percentage Difference Label 1 GT FA': median_percentage_difference_FA,
                    'Avg. HD Epi': avg_hausdorff_epi,
                    'Avg. HD Endo': avg_hausdorff_endo,
                    'F1 Label 1' : f1_original,
                    'Precision':precision,
                    'Recall':recall,   
                    

                }

                # Print each metric on a new line
                print(f'metrics appending....')
                metrics_data.append(metrics_entry)
            else:
                print(f"Original or crop mask not found for {pred_file}. Expected paths:\n - Original MD : {original_image_path_MD}\n - Crop Mask: {crop_mask_path}")
        else:
            print(f"No ground truth found for {pred_file}, skipping.")

        print(f'################################################################################################################################')
        print(f'METRICS:')
        print(f"File: {pred_file}")
        for key, value in metrics_entry.items():
            print(f'{key}: {value}')

        print(f'################################################################################################################################')
    
