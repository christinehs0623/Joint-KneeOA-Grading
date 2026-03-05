import numpy as np
import os
import re # For parsing the .pts file
import matplotlib.pyplot as plt
# import seaborn as sns # For potentially nicer plots
import pandas as pd # Useful for data handling and plotting
import cv2
import pydicom
import math
import h5py


# --- Configuration ---
# Set visit identifier (change this to V00, V01, V03, V05, etc.)
VISIT = "V00"

# Base paths (update base dirs once, works for all visits)
BASE_DIR = "original_data"

# Input paths
IMAGE_DIR = f"{BASE_DIR}/{VISIT}/Bilateral_PA_Fixed_Flexion_Knee"  # Directory containing OAI DICOM 
PTS_DIR = f"{BASE_DIR}/{VISIT}/Points_px"  # Directory containing .pts landmark files from BoneFinder
TXT_FILE_PATH = f"{BASE_DIR}/{VISIT}/kxr_sq_bu{VISIT[1:]}.txt"  # Path to the master TXT file
SHAPELR_NPZ = f"{BASE_DIR}/{VISIT}/id_shapes_LR_{VISIT}.npz"  # Optional: Pre-saved shapes file

# Column names (automatically constructed based on visit)
KL_GRADE_COLUMN = f"{VISIT}XRKL"
JSNM_COLUMN     = f"{VISIT}XRJSM"
JSNL_COLUMN     = f"{VISIT}XRJSL"
OSFM_COLUMN     = f"{VISIT}XROSFM"
OSTM_COLUMN     = f"{VISIT}XROSTM"
OSTL_COLUMN     = f"{VISIT}XROSTL"
OSFL_COLUMN     = f"{VISIT}XROSFL"

# Output HDF5 file (auto named per visit)
# round the number at safeint function
IMG_SIZE = 100
CROP_PATCH_SIZE = 16
OUTPUT_HDF5_PATIENT_GROUPED_FILE = f"{BASE_DIR}/{VISIT}/{VISIT}_knee_patches_patient_grouped_{CROP_PATCH_SIZE }_{IMG_SIZE}_all_feature.h5"

# Other configs
BASE_DICOM_PATH = IMAGE_DIR
TARGET_PATCH_SIZE = (CROP_PATCH_SIZE, CROP_PATCH_SIZE)
PATCH_AREA_PX = IMG_SIZE  * IMG_SIZE  # area in pixels^2 used for patch size scaling


# Use a specific file extension if needed, e.g., "*.png", "*.dcm"
# If filenames *are* the IDs without extensions, use os.listdir directly.
# Assuming filenames might have extensions like .png, .dcm etc.
EXPECTED_LANDMARKS_PER_KNEE = 74
TOTAL_LANDMARKS = EXPECTED_LANDMARKS_PER_KNEE * 2

# --- Provided Function ---
def get_values_by_id(txt_file, search_id, column_name):
    """
    Retrieve two values for SIDE=1 (right) and SIDE=2 (left) where READPRJ=15 for a given ID.
    If READPRJ=15 is not found, fallback to READPRJ=37 or READPRJ=42.

    Parameters:
    - txt_file: numpy array containing the text data.
    - search_id: str, the ID to search for (first column).
    - column_name: str, the name of the column to retrieve value from.

    Returns:
    - Tuple (value_for_SIDE_1, value_for_SIDE_2) or -1 if not found.
    """
    # Extract header row (first row)
    headers = txt_file[0]

    # Find column indices
    # Convert headers to uppercase for case-insensitive checking
    headers_upper = [header.upper() for header in headers] # some headers are in lowercases
    if column_name not in headers_upper:
        print(f"Column '{column_name}' not found!")
        return None, None


    col_index = np.where(np.char.upper(headers) == column_name)[0][0]  # Target column
    side_index = np.where(headers == "SIDE")[0][0]  # 'SIDE' column index
    readprj_index = np.where(np.char.upper(headers) == "READPRJ")[0][0]  # 'READPRJ' column index; some will be in lowercase

    # Try priority READPRJ values in order: 15 → 37 → 42
    readprj_priority = ["15", "37", "42"]
    
    value_1, value_2 = None, None  # Variables to store values

    for readprj in readprj_priority:
        for row in txt_file[1:]:  # Skip header row
            if row[0] == search_id and row[readprj_index] == readprj:  # Match ID and READPRJ
                if row[side_index] == "1":
                    value_1 = row[col_index]
                elif row[side_index] == "2":
                    value_2 = row[col_index]
        
        # If both SIDE=1 and SIDE=2 are found, stop searching
        if value_1 is not None and value_2 is not None:
            if readprj != "15":
                print(f"Using READPRJ={readprj} for ID {search_id} because READPRJ=15 is missing.")
            break
    
    # Handle empty strings and convert to int safely
    # def safe_int(value):
    #     return int(value) if value and value.strip() != '' else None
    
    def safe_int(value):
        if value is None or str(value).strip() == "":
            return None
        try:
            # Try integer first
            return int(value)
        except ValueError:
            try:
                # If not an integer, try float then round
                return round(float(value))
            except ValueError:
                return None

    value_1 = safe_int(value_1)
    value_2 = safe_int(value_2)

    # If any value is still missing, print a warning
    if value_1 is None:
        print(f"Data missing for ID {search_id}: SIDE=1 not found in READPRJ=15, 37, or 42!")
        value_1 = -999
    if value_2 is None:
        print(f"Data missing for ID {search_id}: SIDE=2 not found in READPRJ=15, 37, or 42!")
        value_2 = -999

    return value_1, value_2

    
# --- Function to Read .pts File ---
def read_pts_file(filepath, expected_points):
    """Reads landmark points from a .pts file."""
    points = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Find lines between { and }
        start_index = -1
        end_index = -1
        for i, line in enumerate(lines):
            if line.strip() == '{':
                start_index = i + 1
            elif line.strip() == '}':
                end_index = i
                break
        
        if start_index == -1 or end_index == -1 or start_index >= end_index:
             print(f"Warning: Could not find valid {{ ... }} block in {filepath}")
             return None

        point_lines = lines[start_index:end_index]
        
        for line in point_lines:
            line = line.strip()
            if not line: # Skip empty lines
                continue
            try:
                # Use regex to find floating point numbers, robust to extra spaces
                coords = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", line)
                if len(coords) == 2:
                    points.append([float(coords[0]), float(coords[1])])
                else:
                    print(f"Warning: Skipping malformed line in {filepath}: '{line}'")
            except ValueError:
                print(f"Warning: Could not parse line in {filepath}: '{line}'")
                continue # Skip lines that can't be parsed

        if len(points) != expected_points:
            print(f"Warning: Expected {expected_points} points, but found {len(points)} in {filepath}")
            # Decide how to handle this: return None, return partial data, raise error?
            # Returning None is safer if exact number is critical.
            return None 
            
        return np.array(points) # Shape: (expected_points, 2)

    except FileNotFoundError:
        print(f"Error: PTS file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error reading PTS file {filepath}: {e}")
        return None

# --- Main Processing Logic ---
def main():
    # 1. Get Patient IDs from filenames in IMAGE_DIR
    try:
        # List all items in the directory
        all_files = os.listdir(IMAGE_DIR)
        # Example: Assume filenames are like '9000001.png'. Extract '9000001'.
        # Adjust the splitting logic if your filenames are different.
        patient_ids = sorted([os.path.splitext(f)[0] for f in all_files if os.path.isfile(os.path.join(IMAGE_DIR, f))])
        # Basic check if IDs look reasonable (e.g., numeric) - adjust if needed
        patient_ids = [pid for pid in patient_ids if pid.isdigit()] 
        print(f"Found {len(patient_ids)} potential patient IDs.")
        if not patient_ids:
            raise ValueError("No valid patient IDs found. Check IMAGE_DIR and filename format.")
    except FileNotFoundError:
        print(f"Error: IMAGE_DIR not found: {IMAGE_DIR}")
        patient_ids = []
    except Exception as e:
        print(f"Error listing patient IDs: {e}")
        patient_ids = []
        
    # 2. Load the master TXT file
    try:
        # Use genfromtxt for better handling of potential issues like missing values or mixed types
        # Using delimiter='\t' assuming tab-separated; change if it's space or comma etc.
        # dtype=str to read everything as string initially
        # comments='#' or None if no comment lines expected
        # txt_data = np.genfromtxt(TXT_FILE_PATH, delimiter='\t', dtype=str, skip_header=0, comments=None, invalid_raise=False)
        txt_data = np.loadtxt(TXT_FILE_PATH, dtype=str, delimiter="|", ndmin=2)

        print(f"Successfully loaded TXT data from {TXT_FILE_PATH} with shape {txt_data.shape}")
        if txt_data.ndim == 1: # Handle case where only one row (header) or one data row exists
            txt_data = txt_data.reshape(1, -1) if len(txt_data) > 0 else np.array([[]]) # Reshape or make empty
            print(f"Reshaped TXT data shape: {txt_data.shape}")
        if txt_data.shape[0] < 2:
            print("Warning: TXT data seems to contain only headers or is empty.")
    except FileNotFoundError:
        print(f"Error: TXT file not found: {TXT_FILE_PATH}")
        txt_data = None
    except Exception as e:
        print(f"Error loading TXT file {TXT_FILE_PATH}: {e}")
        txt_data = None

    # 3. Process each patient
    all_patient_features_L = []
    all_patient_features_R = []
    kl_grades_L_list = []
    kl_grades_R_list = []
    processed_ids = []  # To keep track of successfully processed patient IDs

    # Define the auxiliary metric columns you want to extract
    auxiliary_columns = [
        JSNM_COLUMN,
        JSNL_COLUMN,
        OSFM_COLUMN,
        OSTM_COLUMN,
        OSTL_COLUMN,
        OSFL_COLUMN,
    ]

    # Create storage dicts for auxiliary features, separate for left and right knees
    all_patient_aux_features_L = []
    all_patient_aux_features_R = []


    # Assuming txt_data is your DataFrame or dictionary with metric values per patient

    if txt_data is not None and len(patient_ids) > 0:
        for patient_id in patient_ids:
            aux_features_L  = []
            aux_features_R = []
            print(f"\nProcessing patient ID: {patient_id}")

            pts_filepath = os.path.join(PTS_DIR, f"{patient_id}.pts")
            landmarks = read_pts_file(pts_filepath, TOTAL_LANDMARKS)

            if landmarks is None:
                print(f"Skipping patient {patient_id} due to issues reading landmarks.")
                continue

            # Extract KL grades for right and left knees
            kl_right, kl_left = get_values_by_id(txt_data, patient_id, KL_GRADE_COLUMN)

            # Extract auxiliary metric values (right and left) for each column
            for col in auxiliary_columns:
            
                val_right, val_left = get_values_by_id(txt_data, patient_id, col)
                aux_features_R.append(val_right)
                aux_features_L.append(val_left)
            
            print(f"Auxiliary features for {patient_id}: Right={aux_features_R}, Left={aux_features_L}")
                      

            # Flatten landmarks as before
            landmarks_flat = landmarks.flatten()

            # Extract features for left knee landmarks
            feature_vector_L = landmarks_flat[:EXPECTED_LANDMARKS_PER_KNEE*2]
            all_patient_features_L.append(feature_vector_L)
            kl_grades_L_list.append(kl_left)
            all_patient_aux_features_L.append(aux_features_L)

            # Extract features for right knee landmarks
            feature_vector_R = landmarks_flat[EXPECTED_LANDMARKS_PER_KNEE*2:TOTAL_LANDMARKS*2]
            all_patient_features_R.append(feature_vector_R)
            kl_grades_R_list.append(kl_right)
            all_patient_aux_features_R.append(aux_features_R)

            processed_ids.append(patient_id)
            print(f"Successfully processed patient {patient_id}. Feature vector shapes: {feature_vector_L.shape}, {feature_vector_R.shape}")

    print(f"\nCollected features for {len(processed_ids)} patients.")

    all_features_L_np = np.array(all_patient_features_L)
    print(f"Final left knee feature matrix shape: {all_features_L_np.shape}") # (num_patients*2, 150)
    all_features_R_np = np.array(all_patient_features_R)
    print(f"Final right knee feature matrix shape: {all_features_R_np.shape}") # (num_patients*2, 150)

    # Store KL grades separately for convenience in plotting/analysis
    kl_grades_L_np = np.array(kl_grades_L_list) # Shape: (num_patients*2, 1)
    kl_grades_R_np = np.array(kl_grades_R_list) # Shape: (num_patients*2, 1)

    aux_L_np = np.array(all_patient_aux_features_L) # Shape: (num_patients, num_aux_features)
    aux_R_np = np.array(all_patient_aux_features_R) # Shape: (num_patients, num_aux_features)

    shapes_L_2d = transformTo2d(all_features_L_np)
    shapes_R_2d = transformTo2d(all_features_R_np)
    np.savez(SHAPELR_NPZ, id=patient_ids, shapes_L=shapes_L_2d, shapes_R=shapes_R_2d, KL_L=kl_grades_L_np, KL_R=kl_grades_R_np, aux_L_np=aux_L_np, aux_R_np=aux_R_np)
    
    return {
        "id": patient_ids,
        "shapes_L": shapes_L_2d,
        "shapes_R": shapes_R_2d,
        "KL_L": kl_grades_L_np,
        "KL_R": kl_grades_R_np,
        "aux_L_np": aux_L_np,
        "aux_R_np": aux_R_np
    }


def transformTo2d(data):
    '''
    Transform 1d landmarks [x1, y1, x2, y2, ...] into 
    [(x1, y1), 
     (x2, y2), 
       ...   ]
    '''
    if len(data.shape) == 1:
        landmarks = data[:EXPECTED_LANDMARKS_PER_KNEE*2]
        return np.vstack((landmarks[::2], landmarks[1::2])).T

    res = []
    for d in data:
        landmarks = d[:EXPECTED_LANDMARKS_PER_KNEE*2]
        res.append(np.vstack((landmarks[::2], landmarks[1::2])).T)
    return np.array(res)

def patchFromPoint(point, size):
    '''
    point : the landmark
    size  : half of the length of the box (square)
    '''
    topLeft = (int(point[0] - size), int(point[1] - size))
    botRight = (int(point[0] + size), int(point[1] + size))
    return topLeft, botRight

def crop_patch(image, topLeft, botRight):
    x1, y1 = topLeft[0], topLeft[1]
    x2, y2 = botRight[0], botRight[1]
    patch = image[y1:y2, x1:x2]  # Note: (y, x) order for slicing
    return patch

# Function to plot patches in a grid
def plot_patches_grid(patches, title, start_index=0, n_cols=8):
    patches_array = np.stack(patches)
    n_patches = len(patches_array)
    n_rows = math.ceil(n_patches / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for idx, patch in enumerate(patches_array):
        ax = axes[idx]
        ax.imshow(patch, cmap='gray')
        ax.set_title(f"Point {start_index + idx}", fontsize=10)
        ax.axis('off')

    # Hide any unused subplots
    for idx in range(n_patches, n_rows * n_cols):
        axes[idx].axis('off')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Adapt from Tiulpin et al. (2018)
def process_xray(img, cut_min=5, cut_max=99, multiplier=255):
    """
    This function changes the histogram of the image by doing global contrast normalization
    
    Parameters
    ----------
    img : array_like
        Image
    cut_min : int
         Low percentile to trim
    cut_max : int
        Highest percentile trim
    multiplier : int
        Multiplier to apply after global contrast normalization
        
    Returns
    -------
    array_like
        Returns a processed image
    
    """

    img = img.copy()
    lim1, lim2 = np.percentile(img, [cut_min, cut_max])
    img[img < lim1] = lim1
    img[img > lim2] = lim2

    img -= lim1
    img /= img.max()
    img *= multiplier
    return img

def visualize_patches(patient_index, patient_ids, kl_grades_L_np, kl_grades_R_np, shapes_L_2d, shapes_R_2d, base_dir):
    file_path = rf"{base_dir}\{patient_ids[patient_index]}.dcm"
    dicom_data = pydicom.dcmread(file_path)
    print(patient_ids[patient_index], kl_grades_L_np[patient_index], kl_grades_R_np[patient_index])

    # Define intervals for patch extraction (can modify according to your strategy)
    range1 = np.arange(9, 27)
    range2 = np.arange(44, 67)
    interval = np.concatenate([range1, range2])

    # Load image once
    image = dicom_data.pixel_array.copy()
    print(f"Image shape: {image.shape}")

    # Calculate patch size dynamically based on image size
    size = np.sqrt((IMG_SIZE * IMG_SIZE) / (3560 * 4320)) * np.sqrt(image.shape[0]*image.shape[1]) / 2
    print(f"Patch half size: {size}")

    # Draw rectangles on the image for both left and right landmarks
    for i in interval:
        topLeft, botRight = patchFromPoint(shapes_L_2d[patient_index][i], size=size)
        cv2.rectangle(image, topLeft, botRight, color=(255), thickness=1)
        topLeft, botRight = patchFromPoint(shapes_R_2d[patient_index][i], size=size)
        cv2.rectangle(image, topLeft, botRight, color=(255), thickness=1)

    # Plot image with landmarks and patch boxes
    plt.figure(figsize=(35, 43))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(f"{dicom_data.SeriesDescription} ({patient_ids[patient_index]})")
    plt.scatter(shapes_L_2d[patient_index][:, 0], shapes_L_2d[patient_index][:, 1], c="red", label='Left Landmarks')
    plt.scatter(shapes_R_2d[patient_index][:, 0], shapes_R_2d[patient_index][:, 1], c="blue", label='Right Landmarks')
    plt.legend()

    # Annotate landmark indices on the plot
    for i in range(len(shapes_L_2d[patient_index])):
        plt.annotate(i, (shapes_L_2d[patient_index][i, 0], shapes_L_2d[patient_index][i, 1]), color='red')
        plt.annotate(i + len(shapes_L_2d[patient_index]), (shapes_R_2d[patient_index][i, 0], shapes_R_2d[patient_index][i, 1]), color='blue')

    plt.show()


def create_patches_for_knee(processed_image, shapes, point_indices_to_use,
                            patch_side_length, target_size, horizontally_flip=False):
    """Helper function to create patches for a single knee."""
    patches_collected = []
    successful_point_indices = [] # Store indices of points that resulted in a patch

    for point_i in point_indices_to_use:
        if point_i >= len(shapes): # Check if point_i is a valid index for the current patient's shapes
            # print(f"    Warning: Point index {point_i} is out of bounds for shapes (len: {len(shapes)}). Skipping.")
            continue

        center = shapes[point_i]
        topLeft, botRight = patchFromPoint(center, size=patch_side_length)
        patch_raw = crop_patch(processed_image, topLeft, botRight)

        if patch_raw is not None and patch_raw.size > 0:
            resized_patch = cv2.resize(patch_raw, target_size, interpolation=cv2.INTER_AREA)
            if horizontally_flip:
                resized_patch = np.fliplr(resized_patch)
            patches_collected.append(resized_patch)
            successful_point_indices.append(point_i)
    return patches_collected, successful_point_indices

def plot_flipped_vs_right(left_patches, right_patches, interval, n_cols=8):
    # Flip left patches horizontally
    left_patches_flipped = [np.fliplr(p) for p in left_patches]
    
    n_pairs = len(left_patches_flipped)
    n_rows = math.ceil(n_pairs / n_cols)
    
    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(n_cols * 2, n_rows * 4))
    axes = np.array(axes).reshape(n_rows * 2, n_cols)

    for idx in range(n_pairs):
        row = (idx // n_cols) * 2
        col = idx % n_cols
        
        # Top row of each pair: flipped left patch
        axes[row, col].imshow(left_patches_flipped[idx], cmap='gray')
        axes[row, col].set_title(f"L (flipped) {interval[idx]}", fontsize=8)
        axes[row, col].axis('off')
        
        # Bottom row of each pair: right patch
        axes[row + 1, col].imshow(right_patches[idx], cmap='gray')
        axes[row + 1, col].set_title(f"R {interval[idx] + 74}", fontsize=8)
        axes[row + 1, col].axis('off')
    
    plt.tight_layout()
    plt.show()

# === HELPERS ===
def create_patches_for_knee(processed_image, shapes, point_indices_to_use,
                            patch_half_width, target_size, horizontally_flip=False):
    """Create resized patches for one knee."""
    patches_collected = []
    successful_indices = []
    for point_i in point_indices_to_use:
        if point_i >= len(shapes):
            continue
        center = shapes[point_i]
        topLeft, botRight = patchFromPoint(center, size=patch_half_width)
        patch_raw = crop_patch(processed_image, topLeft, botRight)

        if patch_raw is not None and patch_raw.size > 0:
            resized_patch = cv2.resize(patch_raw, target_size, interpolation=cv2.INTER_AREA)
            if horizontally_flip:
                resized_patch = np.fliplr(resized_patch)
            patches_collected.append(resized_patch)
            successful_indices.append(point_i)
    return patches_collected, successful_indices


def save_knee_to_hdf5(hf, patient_id, knee_side, processed_image, shapes, kl_grade,
                      patch_half_width, aux_features, target_size, point_indices, flip=False):
    """
    Extracts patches and saves to HDF5 under patient_id_knee_side group.
    """
    if kl_grade == -999:
        return

    patches, successful_indices = create_patches_for_knee(
        processed_image, shapes, point_indices,
        patch_half_width, target_size, horizontally_flip=flip
    )

    if patches:
        patches_np = np.expand_dims(np.array(patches, dtype=np.float32), axis=-1)
    else:
        patches_np = np.zeros((0, target_size[0], target_size[1], 1), dtype=np.float32)

    group_name = f"{patient_id}_{knee_side}"
    grp = hf.create_group(group_name)
    grp.create_dataset('patches', data=patches_np, compression="gzip")
    grp.create_dataset('kl_grade', data=np.array([kl_grade], dtype=np.int32))
    grp.create_dataset('aux_feature', data=np.array([aux_features], dtype=np.int32))
    grp.create_dataset('patch_source_point_indices', data=np.array(successful_indices, dtype=np.int32))

    # --- Metadata ---
    grp.attrs['side'] = knee_side
    grp.attrs['is_flipped'] = bool(flip)
    grp.attrs['target_patch_size'] = target_size
    grp.attrs['original_num_points'] = len(shapes)
    grp.attrs['requested_point_indices'] = point_indices.tolist()
    grp.attrs['patch_half_width'] = patch_half_width

    print(f"    Stored {patches_np.shape[0]} patches for {group_name} (KL {kl_grade}) with aux features {aux_features}.")


if __name__ == "__main__":
    data = main()
    data = np.load(SHAPELR_NPZ)
    print(data)
    
    patient_ids = data["id"]
    shapes_L_2d = data["shapes_L"]
    shapes_R_2d = data["shapes_R"]
    kl_grades_L_np = data["KL_L"]
    kl_grades_R_np = data["KL_R"]
    aux_features_L = data["aux_L_np"]
    aux_features_R = data["aux_R_np"]
    
    # Set patient index
    index = 0
    print(patient_ids[index], kl_grades_L_np[index], kl_grades_R_np[index], aux_features_L[index], aux_features_R[index])

    # Load DICOM and image
    # file_path = rf"./V00/Bilateral_PA_Fixed_Flexion_Knee/{patient_ids[index]}.dcm"
    file_path = os.path.join(IMAGE_DIR, f"{patient_ids[index]}.dcm")
    dicom_data = pydicom.dcmread(file_path)

    # Preprocess image: convert and normalize (process_xray assumed defined elsewhere)
    image = np.frombuffer(dicom_data.pixel_array.copy(), dtype=np.uint16).copy().astype(np.float64)
    image = process_xray(image, 5, 99, 65535)
    image = image.reshape((dicom_data.Rows, dicom_data.Columns))

    # Calculate patch half-size dynamically based on image resolution and target physical size
    # size = np.sqrt((100 * 100) / (3560 * 4320)) * np.sqrt(image.shape[0] * image.shape[1]) / 2
    # size = np.sqrt((128 * 128) / (3560 * 4320)) * np.sqrt(image.shape[0]*image.shape[1]) / 2
    
    size = np.sqrt((IMG_SIZE * IMG_SIZE) / (3560 * 4320)) * np.sqrt(image.shape[0]*image.shape[1]) / 2

    print(f"Patch half size: {size}")

    # Define indices of landmarks for patches
    range1 = np.arange(9, 27)
    range2 = np.arange(44, 67)
    interval = np.concatenate([range1, range2])

    # Draw rectangles on image for visualization (optional)
    # image_with_boxes = image.copy()
    # for i in interval:
    #     topLeft, botRight = patchFromPoint(shapes_L_2d[index][i], size=size)
    #     cv2.rectangle(image_with_boxes, topLeft, botRight, color=(255), thickness=1)
    #     topLeft, botRight = patchFromPoint(shapes_R_2d[index][i], size=size)
    #     cv2.rectangle(image_with_boxes, topLeft, botRight, color=(255), thickness=1)

    # # Plot the image with rectangles and landmarks
    # plt.figure(figsize=(12, 12))
    # plt.imshow(image_with_boxes, cmap='gray')
    # plt.axis('off')
    # plt.title(dicom_data.SeriesDescription)
    # plt.scatter(shapes_L_2d[index][:, 0], shapes_L_2d[index][:, 1], c='red', label='Left landmarks')
    # plt.scatter(shapes_R_2d[index][:, 0], shapes_R_2d[index][:, 1], c='blue', label='Right landmarks')
    # plt.legend()

    # for i in range(len(shapes_L_2d[index])):
    #     plt.annotate(i, (shapes_L_2d[index][i, 0], shapes_L_2d[index][i, 1]), color='red')
    #     plt.annotate(i + len(shapes_L_2d[index]), (shapes_R_2d[index][i, 0], shapes_R_2d[index][i, 1]), color='blue')
    # plt.show()

    # Prepare to extract and resize patches
    target_size = (16, 16)  # size of each extracted patch (width, height)
    patches_left = []
    patches_right = []

    # Extract patches for left and right knees
    for i in interval:
        # Left knee
        topLeft, botRight = patchFromPoint(shapes_L_2d[index][i], size=size)
        patch = crop_patch(image, topLeft, botRight)
        resized_patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
        patches_left.append(resized_patch)

        # Right knee
        topLeft, botRight = patchFromPoint(shapes_R_2d[index][i], size=size)
        patch = crop_patch(image, topLeft, botRight)
        resized_patch = cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
        patches_right.append(resized_patch)
    
    # Plot left patches
    # plot_patches_grid(patches_left, title='Left Knee Patches', start_index=interval[0])

    # # Plot right patches
    # plot_patches_grid(patches_right, title='Right Knee Patches', start_index=interval[0] + 74)

    # # Plot flipped vs right patches
    # plot_flipped_vs_right(patches_left, patches_right, interval, n_cols=8)


    # Landmark intervals
    range1 = np.arange(9, 27)   # Indices 9..26
    range2 = np.arange(44, 67)  # Indices 44..66
    PATCH_POINT_INDICES = np.concatenate([range1, range2])

    print(f"Processing {len(patient_ids)} patients for grouped HDF5...")
    print(len(shapes_L_2d), len(shapes_R_2d), len(kl_grades_L_np), len(kl_grades_R_np))

    # === MAIN LOOP ===
    with h5py.File(OUTPUT_HDF5_PATIENT_GROUPED_FILE, 'w') as hf:
        # Store global info
        dt = h5py.string_dtype(encoding='utf-8')
        hf.create_dataset('patient_ids_order', data=np.array(patient_ids, dtype=dt))
        hf.create_dataset('global_patch_point_indices_definition', data=PATCH_POINT_INDICES)

        for patient_idx, patient_id_str in enumerate(patient_ids):
            print(f"  Processing patient {patient_idx + 1}/{len(patient_ids)}: {patient_id_str}")

            # --- Load DICOM ---
            file_path = os.path.join(BASE_DICOM_PATH, f"{patient_id_str}.dcm")
            try:
                dicom_data = pydicom.dcmread(file_path)
                image_raw = dicom_data.pixel_array.astype(np.float64)
            except FileNotFoundError:
                print(f"    ERROR: DICOM file not found for {patient_id_str} at {file_path}. Skipping.")
                continue
            except Exception as e:
                print(f"    ERROR: Could not read or process DICOM for {patient_id_str}: {e}. Skipping.")
                continue

            processed_image = process_xray(image_raw, 5, 99, 65535)
            img_h, img_w = image_raw.shape

            # Patch half-width calculation
            patch_half_width = math.sqrt(PATCH_AREA_PX / (3560 * 4320)) * math.sqrt(img_h * img_w) / 2
            # patch_half_width = (np.sqrt(128*128) / np.sqrt(3560*4320)) * np.sqrt(img_h*img_w) / 2

            # Get this patient's shape & KL data
            shapes_L = shapes_L_2d[patient_idx]
            shapes_R = shapes_R_2d[patient_idx]
            kl_L = kl_grades_L_np[patient_idx]
            kl_R = kl_grades_R_np[patient_idx]
            aux_L = aux_features_L[patient_idx]
            aux_R = aux_features_R[patient_idx]

            # --- Save both knees ---
            save_knee_to_hdf5(hf, patient_id_str, "L", processed_image, shapes_L, kl_L,
                            patch_half_width, aux_L, TARGET_PATCH_SIZE, PATCH_POINT_INDICES, flip=True)

            save_knee_to_hdf5(hf, patient_id_str, "R", processed_image, shapes_R, kl_R,
                            patch_half_width, aux_R, TARGET_PATCH_SIZE, PATCH_POINT_INDICES, flip=False)

        print(f"Patient-grouped data saved to {OUTPUT_HDF5_PATIENT_GROUPED_FILE}")