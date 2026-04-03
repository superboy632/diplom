#!/usr/bin/env python3
"""
tf.data Dataset Builder for Crater Detection

This script creates a tf.data.Dataset for training crater detection models.
It loads images and bounding boxes from the converted TensorFlow format dataset
and creates an optimized pipeline with batching, prefetching, and parallel processing.

Author: ML Agent
Date: 2026-03-29
"""

import json
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np

import tensorflow as tf


class CraterDetectionDataset:
    """
    Dataset builder for crater detection using tf.data API.
    
    This class creates optimized TensorFlow datasets for training and validation
    of crater detection models. It handles:
    - Loading images from disk
    - Loading bounding box annotations
    - Data augmentation (optional)
    - Batching and prefetching
    - Efficient parallel processing
    """
    
    def __init__(
        self,
        dataset_path: str,
        image_size: Tuple[int, int] = (416, 416),
        batch_size: int = 16,
        shuffle_buffer_size: int = 1000,
        num_parallel_calls: int = tf.data.AUTOTUNE,
        prefetch_buffer_size: int = tf.data.AUTOTUNE,
        use_augmentation: bool = False,
        subsample_ratio: float = 1.0
    ):
        """
        Initialize the dataset builder.
        
        Args:
            dataset_path: Path to the converted TensorFlow dataset directory
            image_size: Target image size (height, width)
            batch_size: Batch size for training
            shuffle_buffer_size: Buffer size for shuffling
            num_parallel_calls: Number of parallel calls for data loading
            prefetch_buffer_size: Buffer size for prefetching
            use_augmentation: Whether to apply data augmentation
            subsample_ratio: Fraction of data to use (0.0-1.0)
        """
        self.dataset_path = Path(dataset_path)
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_parallel_calls = num_parallel_calls
        self.prefetch_buffer_size = prefetch_buffer_size
        self.use_augmentation = use_augmentation
        self.subsample_ratio = subsample_ratio
        
        # Load dataset metadata
        self.metadata = self._load_metadata()
        self.num_classes = self.metadata['num_classes']
        self.class_names = self.metadata['class_names']
        
        # Load dataset data
        self.dataset_data = self._load_dataset_data()
        
        print(f"Dataset initialized:")
        print(f"  - Image size: {image_size}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Number of classes: {self.num_classes}")
        print(f"  - Subsample ratio: {subsample_ratio}")
        print(f"  - Augmentation: {use_augmentation}")
    
    def _load_metadata(self) -> Dict:
        """
        Load dataset metadata from JSON file.
        
        Returns:
            Dictionary containing dataset metadata
        """
        metadata_file = self.dataset_path / "tf_data_dataset.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Dataset metadata file not found: {metadata_file}")
        
        with open(metadata_file, 'r') as f:
            data = json.load(f)
        
        return data['metadata']
    
    def _load_dataset_data(self) -> Dict:
        """
        Load dataset data from JSON file.
        
        Returns:
            Dictionary containing train and valid splits
        """
        data_file = self.dataset_path / "tf_data_dataset.json"
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data
    
    def _load_image(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        # Construct full path - images are in LU3M6TGT_yolo_format directory
        full_path = str(self.dataset_path.parent / "LU3M6TGT_yolo_format" / image_path)
        
        # Read image file
        image = tf.io.read_file(full_path)
        
        # Decode image (supports PNG, JPEG, etc.)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        
        # Set shape explicitly (required for some operations)
        image.set_shape([None, None, 3])
        
        # Resize to target size
        image = tf.image.resize(image, self.image_size, method='bilinear')
        
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        return image
    
    def _load_bboxes(self, bboxes: List[List[float]]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Load and process bounding boxes.
        
        Args:
            bboxes: List of bounding boxes in [x_min, y_min, x_max, y_max, class_id] format
            
        Returns:
            Tuple of (bbox_coordinates, class_ids)
        """
        if not bboxes:
            # Return empty tensors if no bboxes
            bbox_coords = tf.zeros((0, 4), dtype=tf.float32)
            class_ids = tf.zeros((0,), dtype=tf.int32)
        else:
            # Convert to numpy array first
            bboxes_array = np.array(bboxes, dtype=np.float32)
            
            # Split into coordinates and class IDs
            bbox_coords = bboxes_array[:, :4]  # x_min, y_min, x_max, y_max
            class_ids = bboxes_array[:, 4].astype(np.int32)  # class_id
            
            # Convert to tensors
            bbox_coords = tf.constant(bbox_coords, dtype=tf.float32)
            class_ids = tf.constant(class_ids, dtype=tf.int32)
        
        return bbox_coords, class_ids
    
    def _augment_image(self, image: tf.Tensor, bboxes: tf.Tensor, class_ids: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Apply data augmentation to image and bounding boxes.
        
        Args:
            image: Input image tensor
            bboxes: Bounding box coordinates tensor
            class_ids: Class IDs tensor
            
        Returns:
            Tuple of (augmented_image, augmented_bboxes, class_ids)
        """
        # Random horizontal flip
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            # Flip bounding boxes: x_min -> 1 - x_max, x_max -> 1 - x_min
            width = tf.cast(self.image_size[1], tf.float32)
            x_min = bboxes[:, 0]
            x_max = bboxes[:, 2]
            bboxes = tf.stack([width - x_max, bboxes[:, 1], width - x_min, bboxes[:, 3]], axis=1)
        
        # Random brightness adjustment
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # Random contrast adjustment
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Clip values to valid range
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, bboxes, class_ids
    
    def _data_generator(self, image_paths: List[str], bboxes_list: List[List[List[float]]]):
        """
        Generator function that yields (image, bboxes, class_ids) tuples.
        
        Args:
            image_paths: List of image paths
            bboxes_list: List of bounding box lists
            
        Yields:
            Tuple of (image_tensor, bbox_coords, class_ids)
        """
        for image_path, bboxes in zip(image_paths, bboxes_list):
            # Load image
            full_path = str(self.dataset_path.parent / "LU3M6TGT_yolo_format" / image_path)
            image = tf.io.read_file(full_path)
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image.set_shape([None, None, 3])
            image = tf.image.resize(image, self.image_size, method='bilinear')
            image = tf.cast(image, tf.float32) / 255.0
            
            # Process bboxes
            bbox_coords, class_ids = self._load_bboxes(bboxes)
            
            yield image.numpy(), bbox_coords.numpy(), class_ids.numpy()
    
    def _create_dataset_from_split(
        self,
        split_name: str,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """
        Create a tf.data.Dataset from a specific split.
        
        Args:
            split_name: Name of the split ('train' or 'valid')
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset object
        """
        split_data = self.dataset_data[split_name]
        image_paths = split_data['image_paths']
        bboxes_list = split_data['bboxes']
        
        # Apply subsampling
        if self.subsample_ratio < 1.0:
            num_samples = int(len(image_paths) * self.subsample_ratio)
            indices = np.random.choice(len(image_paths), num_samples, replace=False)
            image_paths = [image_paths[i] for i in indices]
            bboxes_list = [bboxes_list[i] for i in indices]
            print(f"  Subsampled {split_name}: {num_samples} samples (from {len(self.dataset_data[split_name]['image_paths'])})")
        else:
            print(f"  Using all {split_name} samples: {len(image_paths)}")
        
        # Create dataset from generator
        output_signature = (
            tf.TensorSpec(shape=self.image_size + (3,), dtype=tf.float32),
            tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
            tf.TensorSpec(shape=[None], dtype=tf.int32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(image_paths, bboxes_list),
            output_signature=output_signature
        )
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size)
        
        # Apply augmentation if enabled (only for training)
        if self.use_augmentation and split_name == 'train':
            dataset = dataset.map(
                lambda img, bbox, cls: self._augment_image(img, bbox, cls),
                num_parallel_calls=self.num_parallel_calls
            )
        
        # Batch the dataset
        # Note: We use padded_batch to handle variable number of objects per image
        dataset = dataset.padded_batch(
            self.batch_size,
            padded_shapes=(
                self.image_size + (3,),  # image shape
                [None, 4],  # bbox coordinates (variable number of bboxes)
                [None]  # class IDs (variable number of classes)
            ),
            padding_values=(
                0.0,  # image padding
                -1.0,  # bbox padding (use -1 to indicate padding)
                -1  # class ID padding
            )
        )
        
        # Prefetch for performance
        dataset = dataset.prefetch(buffer_size=self.prefetch_buffer_size)
        
        return dataset
    
    def get_train_dataset(self, shuffle: bool = True) -> tf.data.Dataset:
        """
        Get the training dataset.
        
        Args:
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Training tf.data.Dataset
        """
        print("Creating training dataset...")
        return self._create_dataset_from_split('train', shuffle=shuffle)
    
    def get_valid_dataset(self, shuffle: bool = False) -> tf.data.Dataset:
        """
        Get the validation dataset.
        
        Args:
            shuffle: Whether to shuffle the dataset
            
        Returns:
            Validation tf.data.Dataset
        """
        print("Creating validation dataset...")
        return self._create_dataset_from_split('valid', shuffle=shuffle)
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary containing dataset information
        """
        return {
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'train_samples': len(self.dataset_data['train']['image_paths']),
            'valid_samples': len(self.dataset_data['valid']['image_paths']),
            'subsample_ratio': self.subsample_ratio
        }


def create_crater_detection_dataset(
    dataset_path: str = "tensorflow_dataset",
    image_size: Tuple[int, int] = (416, 416),
    batch_size: int = 16,
    shuffle_buffer_size: int = 1000,
    num_parallel_calls: int = tf.data.AUTOTUNE,
    prefetch_buffer_size: int = tf.data.AUTOTUNE,
    use_augmentation: bool = False,
    subsample_ratio: float = 1.0
) -> Tuple[tf.data.Dataset, tf.data.Dataset, CraterDetectionDataset]:
    """
    Convenience function to create crater detection datasets.
    
    Args:
        dataset_path: Path to the converted TensorFlow dataset directory
        image_size: Target image size (height, width)
        batch_size: Batch size for training
        shuffle_buffer_size: Buffer size for shuffling
        num_parallel_calls: Number of parallel calls for data loading
        prefetch_buffer_size: Buffer size for prefetching
        use_augmentation: Whether to apply data augmentation
        subsample_ratio: Fraction of data to use (0.0-1.0)
        
    Returns:
        Tuple of (train_dataset, valid_dataset, dataset_builder)
    """
    # Create dataset builder
    dataset_builder = CraterDetectionDataset(
        dataset_path=dataset_path,
        image_size=image_size,
        batch_size=batch_size,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_calls=num_parallel_calls,
        prefetch_buffer_size=prefetch_buffer_size,
        use_augmentation=use_augmentation,
        subsample_ratio=subsample_ratio
    )
    
    # Create datasets
    train_dataset = dataset_builder.get_train_dataset(shuffle=True)
    valid_dataset = dataset_builder.get_valid_dataset(shuffle=False)
    
    return train_dataset, valid_dataset, dataset_builder


def main():
    """
    Main function to demonstrate dataset creation and usage.
    """
    print("=" * 70)
    print("Crater Detection tf.data Dataset Builder")
    print("=" * 70)
    
    # Configuration
    DATASET_PATH = "tensorflow_dataset"
    IMAGE_SIZE = (416, 416)
    BATCH_SIZE = 16
    SHUFFLE_BUFFER_SIZE = 1000
    USE_AUGMENTATION = True
    SUBSAMPLE_RATIO = 1.0  # Use 0.5 for 50% of data, 1.0 for all data
    
    # Create datasets
    train_dataset, valid_dataset, dataset_builder = create_crater_detection_dataset(
        dataset_path=DATASET_PATH,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle_buffer_size=SHUFFLE_BUFFER_SIZE,
        use_augmentation=USE_AUGMENTATION,
        subsample_ratio=SUBSAMPLE_RATIO
    )
    
    # Print dataset information
    print("\n" + "=" * 70)
    print("Dataset Information:")
    print("=" * 70)
    info = dataset_builder.get_dataset_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Demonstrate iterating through the dataset
    print("\n" + "=" * 70)
    print("Testing Dataset Iteration:")
    print("=" * 70)
    
    # Get one batch from training dataset
    for batch_idx, (images, bboxes, class_ids) in enumerate(train_dataset.take(1)):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Images dtype: {images.dtype}")
        print(f"  Bboxes shape: {bboxes.shape}")
        print(f"  Bboxes dtype: {bboxes.dtype}")
        print(f"  Class IDs shape: {class_ids.shape}")
        print(f"  Class IDs dtype: {class_ids.dtype}")
        
        # Show first sample in batch
        print(f"\n  First sample in batch:")
        print(f"    Image shape: {images[0].shape}")
        print(f"    Number of objects: {tf.reduce_sum(tf.cast(class_ids[0] >= 0, tf.int32)).numpy()}")
        
        # Show bounding boxes for first sample (non-padded)
        valid_mask = class_ids[0] >= 0
        valid_bboxes = bboxes[0][valid_mask]
        valid_classes = class_ids[0][valid_mask]
        
        print(f"    Valid bounding boxes: {len(valid_bboxes)}")
        for i, (bbox, cls) in enumerate(zip(valid_bboxes, valid_classes)):
            print(f"      Object {i+1}: bbox={bbox.numpy()}, class={cls.numpy()}")
    
    print("\n" + "=" * 70)
    print("Dataset created successfully!")
    print("=" * 70)
    print("\nUsage example:")
    print("  train_dataset, valid_dataset, builder = create_crater_detection_dataset()")
    print("  for images, bboxes, class_ids in train_dataset:")
    print("      # Your training code here")
    print("      pass")
    print("=" * 70)


if __name__ == "__main__":
    main()
