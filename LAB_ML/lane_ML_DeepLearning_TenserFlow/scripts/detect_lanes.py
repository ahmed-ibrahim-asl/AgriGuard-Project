# detect_lanes.py

import argparse
import cv2
import numpy as np
import tensorflow as tf
import logging
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Detect lanes in a video using a trained model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model.")
    parser.add_argument('--input_video', type=str, required=True, help="Path to the input video.")
    parser.add_argument('--output_video', type=str, required=True, help="Path to save the output video.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for lane detection.")
    parser.add_argument('--display', action='store_true', help="Display the video during processing.")
    return parser.parse_args()

def preprocess_frame(frame, target_size):
    """
    Preprocess the frame for model prediction.
    - Resize to target_size.
    - Normalize pixel values to [0, 1].
    """
    resized = cv2.resize(frame, target_size)
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0).astype(np.float32)
    return input_tensor

def postprocess_mask(predicted_mask, threshold):
    """
    Post-process the predicted mask.
    - Apply threshold.
    - Convert to binary mask.
    """
    binary_mask = (predicted_mask > threshold).astype(np.uint8)
    binary_mask = binary_mask[:, :, 0]  # Assuming the mask has shape (height, width, 1)
    return binary_mask

def overlay_lanes(frame, mask, color=(0, 0, 255), alpha=0.5):
    """
    Overlay the lane mask onto the original frame.
    - frame: Original frame.
    - mask: Binary mask of detected lanes.
    - color: Color of the lanes (default is red).
    - alpha: Transparency factor for overlay.
    """
    mask_colored = np.zeros_like(frame)
    mask_colored[mask == 1] = color

    combined = cv2.addWeighted(frame, 1, mask_colored, alpha, 0)
    return combined

def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s', level=logging.INFO)

    # Load the model
    logging.info(f"Loading model from {args.model_path}")
    model = tf.keras.models.load_model(args.model_path, compile=False)

    # Define target size (should match the model's expected input size)
    target_size = (1280, 720)  # (width, height)

    # Open the video file
    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        logging.error(f"Error opening video file {args.input_video}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logging.info(f"Frame Width: {frame_width}, Frame Height: {frame_height}, FPS: {fps}")
    logging.info(f"Output Video will be saved to {args.output_video}")
    logging.info(f"Total Frames to Process: {total_frames}")

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_video, fourcc, fps, (frame_width, frame_height))

    frame_idx = 0

    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Frame read failed at index {frame_idx}. Skipping...")
            break

        # Preprocess the frame
        input_tensor = preprocess_frame(frame, target_size)

        # Predict the mask
        predicted_mask = model.predict(input_tensor, verbose=0)[0]  # Shape: (height, width, 1)

        # Log statistics for debugging
        logging.info(f"Frame {frame_idx}: Predicted mask min={predicted_mask.min()}, max={predicted_mask.max()}, mean={predicted_mask.mean()}")

        # Save predicted mask for inspection (optional)
        if frame_idx < 5:
            cv2.imwrite(f'predicted_mask_frame_{frame_idx}.png', predicted_mask[:, :, 0] * 255)

        # Post-process the mask
        binary_mask = postprocess_mask(predicted_mask, args.threshold)

        # Resize mask to original frame size
        binary_mask_resized = cv2.resize(binary_mask, (frame_width, frame_height), interpolation=cv2.INTER_NEAREST)

        # Overlay the lanes on the original frame
        combined_frame = overlay_lanes(frame, binary_mask_resized)

        # Write the frame to the output video
        out.write(combined_frame)

        # Display the frame if requested
        if args.display:
            cv2.imshow('Lane Detection', combined_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logging.info("Display window closed by user.")
                break

        # Release memory
        del input_tensor, predicted_mask, binary_mask, binary_mask_resized, combined_frame

        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    if args.display:
        cv2.destroyAllWindows()

    logging.info("Processing completed successfully.")

if __name__ == "__main__":
    main()
