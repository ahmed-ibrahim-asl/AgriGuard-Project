# scripts/train_model.py

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import argparse
import logging
from tqdm import tqdm

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_args():
    parser = argparse.ArgumentParser(description="Train a U-Net model for lane detection.")
    parser.add_argument('--train_images_dir', type=str, default=None, help="Path to training images directory.")
    parser.add_argument('--train_masks_dir', type=str, default=None, help="Path to training masks directory.")
    parser.add_argument('--val_images_dir', type=str, default=None, help="Path to validation images directory.")
    parser.add_argument('--val_masks_dir', type=str, default=None, help="Path to validation masks directory.")
    parser.add_argument('--model_save_path', type=str, default=None, help="Path to save the trained model.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size for training (default: 4).")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Initial learning rate (default: 1e-4).")
    parser.add_argument('--img_width', type=int, default=1280, help="Width to resize images (default: 1280).")
    parser.add_argument('--img_height', type=int, default=720, help="Height to resize images (default: 720).")
    return parser.parse_args()

def build_unet_model(input_size=(720, 1280, 3)):
    inputs = tf.keras.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

def load_data(images_dir, masks_dir, img_width, img_height):
    image_files = sorted([
        f for f in os.listdir(images_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    images = []
    masks = []

    for img_file in tqdm(image_files, desc="Loading images and masks"):
        # Load and preprocess image
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logging.warning(f"Failed to load image {img_path}. Skipping.")
            continue
        img = cv2.resize(img, (img_width, img_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255.0  # Normalize to [0,1]
        images.append(img)

        # Load and preprocess mask
        mask_file = f"{os.path.splitext(img_file)[0]}_label.png"
        mask_path = os.path.join(masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            logging.warning(f"Failed to load mask {mask_path}. Skipping corresponding image.")
            images.pop()  # Remove the last image since mask is missing
            continue
        mask = cv2.resize(mask, (img_width, img_height))
        mask = mask / 255.0  # Normalize to [0,1]
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        masks.append(mask)

    images = np.array(images, dtype=np.float32)
    masks = np.array(masks, dtype=np.float32)

    logging.info(f"Loaded {len(images)} images and masks.")
    return images, masks

def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

def display_device_info():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        logging.info(f"TensorFlow is using GPU: {physical_devices[0].name}")
    else:
        logging.info("TensorFlow is using CPU.")

def main():
    setup_logging()
    args = parse_args()

    # Configure GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            logging.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            logging.error(e)
    else:
        logging.info("No GPU found. TensorFlow is using CPU.")

    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define paths relative to the project root
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    train_images_dir = args.train_images_dir if args.train_images_dir else os.path.join(project_root, 'data', 'processed', 'train', 'images')
    train_masks_dir = args.train_masks_dir if args.train_masks_dir else os.path.join(project_root, 'data', 'processed', 'train', 'masks')
    val_images_dir = args.val_images_dir if args.val_images_dir else os.path.join(project_root, 'data', 'processed', 'val', 'images')
    val_masks_dir = args.val_masks_dir if args.val_masks_dir else os.path.join(project_root, 'data', 'processed', 'val', 'masks')
    model_save_path = args.model_save_path if args.model_save_path else os.path.join(project_root, 'models', 'best_model.keras')  # Changed to .keras

    img_width = args.img_width
    img_height = args.img_height

    # Create models directory if it doesn't exist
    models_dir = os.path.dirname(model_save_path)
    os.makedirs(models_dir, exist_ok=True)

    # Load training data
    logging.info("Loading training data...")
    train_images, train_masks = load_data(train_images_dir, train_masks_dir, img_width, img_height)

    # Load validation data
    logging.info("Loading validation data...")
    val_images, val_masks = load_data(val_images_dir, val_masks_dir, img_width, img_height)

    if len(train_images) == 0 or len(val_images) == 0:
        logging.error("Insufficient data for training or validation. Exiting.")
        return

    # Build model
    logging.info("Building U-Net model...")
    model = build_unet_model(input_size=(img_height, img_width, 3))
    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy', iou_metric])

    # Define callbacks
    checkpoint = ModelCheckpoint(
        model_save_path,
        verbose=1,
        save_best_only=True,
        monitor='val_iou_metric',
        mode='max'
        # Removed 'save_format' parameter
    )
    early_stop = EarlyStopping(monitor='val_iou_metric', patience=10, mode='max', verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_iou_metric', factor=0.5, patience=5, mode='max', verbose=1)

    callbacks = [checkpoint, early_stop, reduce_lr]

    # Summary of the model
    logging.info("Model Summary:")
    model.summary(print_fn=lambda x: logging.info(x))

    # Train the model
    logging.info("Starting model training...")
    history = model.fit(
        train_images,
        train_masks,
        validation_data=(val_images, val_masks),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    logging.info(f"Training completed. Best model saved at {model_save_path}")

if __name__ == "__main__":
    main()
