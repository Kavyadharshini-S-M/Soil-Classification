#refining edit 2.py
#this one is for soil segmentation using color and morphology
#this one is better than edit2.py
#this is the final version of soil segmentation
#working

import cv2
import os
import numpy as np

input_root = "dataset" # Your folder with raw images
output_root = "segmented_soil_dataset_old" # Where to save processed images

valid_exts = ['.jpg', '.jpeg', '.png']

def is_image(file):
    return os.path.splitext(file)[1].lower() in valid_exts

def segment_soil(img_path, save_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load image: {img_path}")
        return False

    # --- 1. Preprocessing ---
    original_for_bitwise = img.copy() # Keep original colors for final bitwise_and
    # Resize for consistent processing and manageable contour sizes
    # Choose a size that balances detail and processing speed
    TARGET_WIDTH = 600 # Or 400, or 800 - experiment
    aspect_ratio = img.shape[0] / img.shape[1]
    target_height = int(TARGET_WIDTH * aspect_ratio)
    
    img_resized = cv2.resize(img, (TARGET_WIDTH, target_height))
    original_for_bitwise = cv2.resize(original_for_bitwise, (TARGET_WIDTH, target_height))

    # Gaussian blur to reduce noise and smooth textures slightly
    blurred_img = cv2.GaussianBlur(img_resized, (7, 7), 0)

    # --- 2. Color Segmentation (HSV) ---
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # --- DEFINE SOIL COLOR RANGES (CRUCIAL - NEEDS TUNING!) ---
    # These are examples and WILL need adjustment for your specific soil types.
    # You might need multiple ranges if your soil colors are very diverse.

    # Example for brownish soils:
    # Hue: Browns can range from orange-ish (around 10-20) to reddish (0-10 and 170-179)
    # Saturation: Soil is usually not super saturated. Avoid very low saturation (greys) unless black soil.
    # Value: Avoid very dark (shadows) or very bright (highlights) if they are problematic.

    # Range 1: General Brown/Dark Soil (covers many cases)
    # Hue: 0-30 (reds to oranges/yellows) - brown often falls here
    # Saturation: 20-255 (avoiding very desaturated grey/whites unless black soil)
    # Value: 20-220 (avoiding pure black and pure white highlights)
    lower_soil1 = np.array([0, 20, 20])
    upper_soil1 = np.array([40, 255, 230]) # Broadened Hue a bit, higher Value for lighter browns
    mask1 = cv2.inRange(hsv_img, lower_soil1, upper_soil1)

    # Range 2: For very dark/blackish soil (low saturation, low value)
    # Hue: 0-179 (any hue, as black has little color)
    # Saturation: 0-80 (low saturation)
    # Value: 0-70 (low value/brightness)
    lower_soil2 = np.array([0, 0, 0])
    upper_soil2 = np.array([179, 100, 90]) # Increased S and V slightly for dark browns
    mask2 = cv2.inRange(hsv_img, lower_soil2, upper_soil2)

    # Combine masks if you use multiple ranges
    # soil_mask_hsv = cv2.bitwise_or(mask1, mask2)
    soil_mask_hsv = mask1 # Start with one, then add more if needed.
                          # If mask2 is too aggressive, it might pick up shadows.

    # --- Optional: Adaptive Thresholding (as a secondary measure or alternative) ---
    # gray_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2GRAY)
    # adaptive_thresh_mask = cv2.adaptiveThreshold(gray_img, 255,
    #                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                           cv2.THRESH_BINARY_INV, 25, 7) # Tune params
    # Combine: soil_mask = cv2.bitwise_and(soil_mask_hsv, adaptive_thresh_mask)
    # Or just use one:
    primary_mask = soil_mask_hsv


    # --- 3. Morphological Operations to clean the mask ---
    # Kernel size depends on the image resolution and noise level
    kernel_size_close = 7 # Larger kernel for closing to fill bigger gaps in soil
    kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
    closed_mask = cv2.morphologyEx(primary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)

    kernel_size_open = 5 # Smaller kernel for opening to remove small noise
    kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    final_binary_mask = opened_mask

    # --- 4. Contour Processing to find the main soil mass ---
    contours, hierarchy = cv2.findContours(final_binary_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to draw the selected soil contours
    soil_contour_mask = np.zeros_like(final_binary_mask)

    if contours:
        # Filter contours by area. This is more robust than just "largest".
        # Adjust min_area based on your image size and expected soil sample size.
        # At TARGET_WIDTH=600, an area of 10000 might be reasonable for a significant soil portion.
        min_contour_area = (TARGET_WIDTH * target_height) * 0.05 # e.g., 5% of total image area
        significant_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_contour_area:
                significant_contours.append(cnt)
        
        if significant_contours:
            # Draw all significant contours onto the mask
            cv2.drawContours(soil_contour_mask, significant_contours, -1, 255, -1) # -1 to draw all
        else:
            # If no significant contours after HSV and morph ops,
            # it might mean the soil fills the frame and HSV picked it all up,
            # OR the HSV ranges are too restrictive.
            # In this case, we can assume the 'final_binary_mask' (after morph ops) IS the soil mask
            # if it's mostly white (i.e., soil fills the frame and was selected by HSV).
            # Check if final_binary_mask is mostly white
            if np.sum(final_binary_mask > 0) > 0.7 * final_binary_mask.size: # 70% of pixels are white
                 soil_contour_mask = final_binary_mask
            else:
                print(f"No significant soil contours found for {img_path} after filtering.")
                # Fallback: if segmentation fails badly, maybe save original or black image
                # For now, we'll proceed with an empty mask, resulting in a black image
                # Or, you could use the original_for_bitwise here.

    else:
        print(f"No contours found at all for {img_path}")
        # Fallback for no contours at all.

    # --- 5. Apply the Mask ---
    result_img = cv2.bitwise_and(original_for_bitwise, original_for_bitwise, mask=soil_contour_mask)

    # --- Optional: Visualization for Debugging (run on a single image) ---
    # cv2.imshow("1. Resized", img_resized)
    # cv2.imshow("2. HSV Mask (Raw)", primary_mask)
    # cv2.imshow("3. Morph Closed", closed_mask)
    # cv2.imshow("4. Morph Opened (Final Binary)", final_binary_mask)
    # cv2.imshow("5. Soil Contour Mask", soil_contour_mask)
    # cv2.imshow("6. Result", result_img)
    # if cv2.waitKey(0) & 0xFF == ord('q'):
    #     cv2.destroyAllWindows()
    # return # Stop after one image for debugging

    # --- 6. Save the Result ---
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, result_img)
    # print(f"Processed: {img_path} -> {save_path}")
    return True


# --- Main Loop ---
print(f"Starting segmentation from '{input_root}' to '{output_root}'\n")
processed_count = 0
failed_count = 0
total_images = 0

# First, count total images for progress
for category_type in os.listdir(input_root):
    type_path = os.path.join(input_root, category_type)
    if not os.path.isdir(type_path): continue
    for class_name in os.listdir(type_path):
        class_path = os.path.join(type_path, class_name)
        if not os.path.isdir(class_path): continue
        for img_file in os.listdir(class_path):
            if is_image(img_file):
                total_images += 1
print(f"Total images to process: {total_images}")

current_image_num = 0
for category_type in os.listdir(input_root): # e.g., "Training", "Validation"
    type_path = os.path.join(input_root, category_type)
    if not os.path.isdir(type_path):
        continue

    # print(f"Processing category: {category_type}")
    for class_name in os.listdir(type_path): # e.g., "Sandy", "Clay"
        class_path = os.path.join(type_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # print(f"  Processing class: {class_name}")
        img_files_in_class = [f for f in os.listdir(class_path) if is_image(f)]

        for img_file in img_files_in_class:
            current_image_num += 1
            input_image_path = os.path.join(class_path, img_file)
            relative_path_from_input_root = os.path.relpath(input_image_path, input_root)
            output_image_path = os.path.join(output_root, relative_path_from_input_root)
            
            if segment_soil(input_image_path, output_image_path):
                processed_count += 1
            else:
                failed_count += 1
            
            if current_image_num % 20 == 0 or current_image_num == total_images:
                print(f"Progress: {current_image_num}/{total_images} images processed. ({processed_count} succeeded, {failed_count} failed)")


print(f"\nSegmentation finished.")
print(f"Successfully processed: {processed_count} images.")
print(f"Failed to process: {failed_count} images.")
print(f"Segmented images saved in '{output_root}'")