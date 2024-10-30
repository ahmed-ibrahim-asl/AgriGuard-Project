# scripts/split_dataset.py

import os
import shutil
from sklearn.model_selection import train_test_split
import argparse
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset to include in the validation split (default: 0.2).")
    parser.add_argument('--random_state', type=int, default=42, help="Random state for reproducibility (default: 42).")
    return parser.parse_args()

def split_dataset(test_size=0.2, random_state=42):
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    images_dir = os.path.join(project_root, 'data', 'lane_images')
    annotations_dir = os.path.join(project_root, 'data', 'annotations')
    processed_dir = os.path.join(project_root, 'data', 'processed')

    # Define train and validation directories
    train_dir = os.path.join(processed_dir, 'train')
    val_dir = os.path.join(processed_dir, 'val')

    # Log the resolved paths
    logging.info(f"Project root: {project_root}")
    logging.info(f"Images directory: {images_dir}")
    logging.info(f"Annotations directory: {annotations_dir}")
    logging.info(f"Processed directory: {processed_dir}")

    # Check if images_dir exists
    if not os.path.exists(images_dir):
        logging.error(f"The images directory does not exist at {images_dir}. Please create it and add your lane images.")
        return

    # Check if annotations_dir exists
    if not os.path.exists(annotations_dir):
        logging.error(f"The annotations directory does not exist at {annotations_dir}. Please create it and add your annotation masks.")
        return

    # Create processed directories
    for split in ['train/images', 'train/masks', 'val/images', 'val/masks']:
        split_path = os.path.join(processed_dir, split)
        os.makedirs(split_path, exist_ok=True)
        logging.info(f"Ensured existence of directory: {split_path}")

    # Get list of image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        logging.error(f"No image files found in {images_dir}. Please add images to annotate.")
        return

    logging.info(f"Found {len(image_files)} image files.")

    # Optional: Filter out images without corresponding masks
    filtered_image_files = [
        f for f in image_files
        if os.path.exists(os.path.join(annotations_dir, f"{os.path.splitext(f)[0]}_label.png"))
    ]

    missing_masks = set(image_files) - set(filtered_image_files)
    if missing_masks:
        for img_file in missing_masks:
            logging.warning(f"Mask for {img_file} does not exist. It will be skipped.")
    image_files = filtered_image_files

    if not image_files:
        logging.error("No images with corresponding masks found. Exiting.")
        return

    logging.info(f"{len(image_files)} images have corresponding masks and will be processed.")

    # Split into train and validation
    train_files, val_files = train_test_split(
        image_files,
        test_size=test_size,
        random_state=random_state
    )

    logging.info(f"Split {len(train_files)} images into training set.")
    logging.info(f"Split {len(val_files)} images into validation set.")

    # Function to copy files
    def copy_files(file_list, split):
        for img_file in file_list:
            # Source and destination for images
            src_img = os.path.join(images_dir, img_file)
            dst_img = os.path.join(processed_dir, split, 'images', img_file)
            shutil.copy(src_img, dst_img)
            logging.debug(f"Copied image {src_img} to {dst_img}")

            # Source and destination for masks
            mask_file = f"{os.path.splitext(img_file)[0]}_label.png"
            src_mask = os.path.join(annotations_dir, mask_file)
            dst_mask = os.path.join(processed_dir, split, 'masks', mask_file)
            if os.path.exists(src_mask):
                shutil.copy(src_mask, dst_mask)
                logging.debug(f"Copied mask {src_mask} to {dst_mask}")
            else:
                logging.warning(f"Mask {mask_file} does not exist. Skipping.")

    # Copy files to training set
    logging.info("Copying training files...")
    copy_files(train_files, 'train')

    # Copy files to validation set
    logging.info("Copying validation files...")
    copy_files(val_files, 'val')

    logging.info("Dataset split into training and validation sets successfully.")

if __name__ == "__main__":
    setup_logging()
    args = parse_args()
    split_dataset(test_size=args.test_size, random_state=args.random_state)
