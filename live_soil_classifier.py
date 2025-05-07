import cv2
import numpy as np
import joblib
import os
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern # Ensure scikit-image is installed

# --- Configuration (MATCH TRAINING CONFIGURATION) ---
IMAGE_SIZE_TRAINING = (128, 128) # The size your features were extracted from during training
GLCM_DISTANCES_TRAINING = [1, 3, 5]
GLCM_ANGLES_TRAINING = [0, np.pi/4, np.pi/2, 3*np.pi/4]
LBP_RADIUS_TRAINING = 3
LBP_POINTS_TRAINING = 8 * LBP_RADIUS_TRAINING

# --- Paths to Saved Model Artifacts ---
MODEL_DIR = "trained_model_old" # Folder where you saved the .joblib files
MODEL_PATH = os.path.join(MODEL_DIR, "soil_classifier_model_randomforest.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.1joblib")

# --- 1. Segmentation Function (COPY EXACTLY FROM YOUR SEGMENTATION SCRIPT) ---
# This function needs to be identical to the one used to create 'segmented_soil_dataset'
# For brevity, I'll sketch it. Make sure to copy your full `segment_soil` function here.
def segment_live_frame(frame):
    # --- Paste your full segment_soil function's logic here ---
    # adapting it to take a 'frame' (numpy array) instead of 'img_path'
    # and return the segmented image (numpy array) and a success flag or the mask itself.

    img_resized = cv2.resize(frame, (600, int(600 * frame.shape[0]/frame.shape[1]))) # Or your chosen TARGET_WIDTH
    original_for_bitwise = img_resized.copy()
    blurred_img = cv2.GaussianBlur(img_resized, (7, 7), 0)
    hsv_img = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)

    # --- USE THE SAME HSV RANGES AS IN YOUR DATASET SEGMENTATION ---
    lower_soil1 = np.array([0, 20, 20]) # EXAMPLE - USE YOUR TUNED VALUES
    upper_soil1 = np.array([40, 255, 230]) # EXAMPLE - USE YOUR TUNED VALUES
    # ... (add other ranges if you used them)
    soil_mask_hsv = cv2.inRange(hsv_img, lower_soil1, upper_soil1)
    # ... (combine masks if needed) ...
    primary_mask = soil_mask_hsv

    kernel_size_close = 7
    kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
    closed_mask = cv2.morphologyEx(primary_mask, cv2.MORPH_CLOSE, kernel_close, iterations=3)
    kernel_size_open = 5
    kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
    final_binary_mask_for_contours = opened_mask

    contours, _ = cv2.findContours(final_binary_mask_for_contours.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    soil_contour_mask = np.zeros_like(final_binary_mask_for_contours)

    if contours:
        min_contour_area = (img_resized.shape[1] * img_resized.shape[0]) * 0.05 # Example
        significant_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
        if significant_contours:
            cv2.drawContours(soil_contour_mask, significant_contours, -1, 255, -1)
        elif np.sum(final_binary_mask_for_contours > 0) > 0.7 * final_binary_mask_for_contours.size:
             soil_contour_mask = final_binary_mask_for_contours


    # Important: The 'result_img' is what your feature extractor will see
    result_img = cv2.bitwise_and(original_for_bitwise, original_for_bitwise, mask=soil_contour_mask)
    
    # Return the segmented image (that will be fed to feature extraction)
    # and the mask for visualization (optional)
    return result_img, soil_contour_mask


# --- 2. Feature Extraction Functions (MATCH TRAINING SCRIPT) ---
def extract_color_histogram(image, bins=(8, 8, 8)): # Ensure bins match training
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Handle cases where image might be all black after segmentation
    if np.sum(image) == 0: # All black image
        return np.zeros(bins[0]*bins[1]*bins[2]) # Or appropriate sized zero vector

    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_haralick_textures(gray_image):
    if np.sum(gray_image) == 0: # All black image
        props_to_extract = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        expected_len = len(props_to_extract) * len(GLCM_DISTANCES_TRAINING) * len(GLCM_ANGLES_TRAINING)
        return np.zeros(expected_len)
    try:
        glcm = graycomatrix(gray_image, distances=GLCM_DISTANCES_TRAINING, angles=GLCM_ANGLES_TRAINING, symmetric=True, normed=True)
        props_to_extract = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        haralick_features = []
        for prop in props_to_extract:
            haralick_features.extend(graycoprops(glcm, prop).ravel())
        return np.array(haralick_features)
    except Exception as e:
        # print(f"Haralick error: {e}") # For debugging
        props_to_extract = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        expected_len = len(props_to_extract) * len(GLCM_DISTANCES_TRAINING) * len(GLCM_ANGLES_TRAINING)
        return np.zeros(expected_len)


def extract_lbp_features(gray_image):
    if np.sum(gray_image) == 0: # All black image
         return np.zeros(LBP_POINTS_TRAINING + 2)

    lbp = local_binary_pattern(gray_image, P=LBP_POINTS_TRAINING, R=LBP_RADIUS_TRAINING, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS_TRAINING + 3), range=(0, LBP_POINTS_TRAINING + 2))
    hist = hist.astype("float")
    hist_sum = hist.sum()
    if hist_sum > 0: # Avoid division by zero
        hist /= hist_sum
    else: # if hist_sum is zero, lbp image was likely all zeros
        hist = np.zeros_like(hist)
    return hist

def extract_features_from_frame(frame_processed): # frame_processed is the segmented image
    # Resize to the size used for feature extraction during training
    image_for_features = cv2.resize(frame_processed, IMAGE_SIZE_TRAINING)

    color_hist = extract_color_histogram(image_for_features)

    gray = cv2.cvtColor(image_for_features, cv2.COLOR_BGR2GRAY)
    # No need for blur here if training features didn't use an extra blur at this stage

    haralick = extract_haralick_textures(gray)
    lbp = extract_lbp_features(gray)

    global_features = np.hstack([color_hist, haralick, lbp])
    return global_features

# --- 3. Load Model and Helpers ---
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Model, scaler, and label encoder loaded successfully.")
except FileNotFoundError:
    print("ERROR: Model files not found. Make sure they are in the 'trained_model' directory.")
    print(f"Looking for: {MODEL_PATH}, {SCALER_PATH}, {LABEL_ENCODER_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model artifacts: {e}")
    exit()


# --- 4. Main Camera Loop ---
cap = cv2.VideoCapture(0) # 0 for default camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("\nStarting live classification. Press 'q' to quit.")
prediction_text = "Initializing..."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    display_frame = frame.copy() # For drawing text and boxes

    # --- A. Segment the Live Frame ---
    # The `segment_live_frame` should return the image that features will be extracted from
    # and optionally a mask for display
    segmented_frame_for_features, segmentation_display_mask = segment_live_frame(frame.copy())

    if segmented_frame_for_features is None:
        cv2.putText(display_frame, "Segmentation Failed", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('Live Soil Classification', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # --- B. Extract Features from the Segmented Frame ---
    # Ensure the segmented_frame_for_features is what your `extract_features_from_frame` expects
    live_features = extract_features_from_frame(segmented_frame_for_features)

    if live_features is None : # Should not happen if extract_features_from_frame handles all-black
        prediction_text = "Feature Extraction Failed"
    elif live_features.shape[0] != model.n_features_in_: # Check feature dimension
         prediction_text = f"Feat. dim mismatch: {live_features.shape[0]} vs {model.n_features_in_}"
    else:
        # --- C. Scale Features and Predict ---
        live_features_scaled = scaler.transform(live_features.reshape(1, -1))
        prediction_encoded = model.predict(live_features_scaled)
        prediction_proba = model.predict_proba(live_features_scaled)

        predicted_label = label_encoder.inverse_transform(prediction_encoded)[0]
        confidence = np.max(prediction_proba) * 100
        prediction_text = f"{predicted_label} ({confidence:.2f}%)"


    # --- D. Display Results ---
    cv2.putText(display_frame, prediction_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Optionally display the segmented view as well
    # Ensure segmentation_display_mask is 3-channel if you want to overlay it or show side-by-side
    if segmentation_display_mask is not None and segmentation_display_mask.ndim == 2:
         segmentation_display_mask_color = cv2.cvtColor(segmentation_display_mask, cv2.COLOR_GRAY2BGR)
         # Concatenate for side-by-side view, make sure heights match
         # combined_view = np.hstack((display_frame, segmentation_display_mask_color))
         # cv2.imshow('Live Soil Classification & Segmentation', combined_view)
         # Or just show the main frame with text
         cv2.imshow('Live Soil Classification', display_frame)
         cv2.imshow('Segmentation Mask', segmentation_display_mask) # Show mask separately
         cv2.imshow('Segmented for Features', segmented_frame_for_features) # What goes into feature extractor

    else:
        cv2.imshow('Live Soil Classification', display_frame)


    if cv2.waitKey(30) & 0xFF == ord('q'): # Increased waitKey for smoother display
        break

cap.release()
cv2.destroyAllWindows()