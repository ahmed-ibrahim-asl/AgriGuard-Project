# scripts/annotate_images.py

import cv2
import os
import numpy as np
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)

# Global variables
drawing = False  # True if the mouse is pressed
current_line = []
img = None
mask = None

def draw_line(event, x, y, flags, param):
    global drawing, current_line, img, mask

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_line = [(x, y)]
        logging.info(f"Started drawing at: ({x}, {y})")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(img, current_line[-1], (x, y), (0, 0, 255), 2)
            cv2.line(mask, current_line[-1], (x, y), 255, 2)
            current_line.append((x, y))

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(img, current_line[-1], (x, y), (0, 0, 255), 2)
        cv2.line(mask, current_line[-1], (x, y), 255, 2)
        logging.info(f"Finished drawing at: ({x}, {y})")

def parse_args():
    parser = argparse.ArgumentParser(description="Annotate lane images.")
    parser.add_argument('--images_dir', type=str, default=None, help="Path to the lane images directory.")
    parser.add_argument('--annotations_dir', type=str, default=None, help="Path to save annotations.")
    parser.add_argument('--image_width', type=int, default=1280, help="Width to resize images.")
    parser.add_argument('--image_height', type=int, default=720, help="Height to resize images.")
    return parser.parse_args()

def annotate_images(images_dir, annotations_dir, img_width=1280, img_height=720):
    global img, mask

    # Verify that images_dir exists
    if not os.path.exists(images_dir):
        logging.error(f"The images directory does not exist at {images_dir}. Please create it and add your lane images.")
        return

    # Get list of image files
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if not image_files:
        logging.error(f"No images found in {images_dir}. Please add images to annotate.")
        return

    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to load {img_path}")
            continue
        img = cv2.resize(img, (img_width, img_height))
        mask = np.zeros((img_height, img_width), dtype=np.uint8)

        cv2.namedWindow('Image')
        cv2.setMouseCallback('Image', draw_line)

        logging.info(f"\nAnnotating {img_file}:")
        logging.info("Instructions:")
        logging.info(" - Click and drag the left mouse button to draw lane lines.")
        logging.info(" - Press 's' to save the annotation.")
        logging.info(" - Press 'c' to clear the current annotations.")
        logging.info(" - Press 'n' to move to the next image.\n")

        while True:
            cv2.imshow('Image', img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):  # Next image
                logging.info("Moving to the next image.")
                break
            elif key == ord('s'):  # Save annotation
                mask_filename = os.path.splitext(img_file)[0] + '_label.png'
                mask_path = os.path.join(annotations_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
                logging.info(f"Saved annotation to {mask_path}")
            elif key == ord('c'):  # Clear annotations for the current image
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (img_width, img_height))
                    mask = np.zeros((img_height, img_width), dtype=np.uint8)
                    logging.info("Cleared annotations. You can start annotating again.")
                else:
                    logging.warning(f"Failed to reload {img_path}. Skipping to next image.")
                    break

        cv2.destroyAllWindows()

def main():
    args = parse_args()

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define absolute paths based on script location and command-line arguments
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    default_images_dir = os.path.join(project_root, 'data', 'lane_images')
    default_annotations_dir = os.path.join(project_root, 'data', 'annotations')

    images_dir = os.path.abspath(args.images_dir) if args.images_dir else default_images_dir
    annotations_dir = os.path.abspath(args.annotations_dir) if args.annotations_dir else default_annotations_dir

    # Ensure annotations directory exists
    os.makedirs(annotations_dir, exist_ok=True)
    logging.info(f"Images directory: {images_dir}")
    logging.info(f"Annotations directory: {annotations_dir}")

    annotate_images(images_dir, annotations_dir, img_width=args.image_width, img_height=args.image_height)

if __name__ == "__main__":
    main()
