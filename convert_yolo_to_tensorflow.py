#!/usr/bin/env python3
"""
YOLO to TensorFlow Dataset Converter

This script converts YOLO-format labels to TensorFlow-compatible bounding box format.
It reads YOLO label files and converts them to (x_min, y_min, x_max, y_max, class_id)
format suitable for TensorFlow object detection pipelines.

Author: ML Agent
Date: 2026-03-29
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np


class YOLOToTensorFlowConverter:
    """
    Converter class for transforming YOLO labels to TensorFlow format.
    
    YOLO Format:
        - Each line: class_id center_x center_y width height
        - All values are normalized (0-1)
        - center_x, center_y: center of bounding box
        - width, height: dimensions of bounding box
    
    TensorFlow Format:
        - Bounding boxes: [x_min, y_min, x_max, y_max, class_id]
        - Coordinates in pixel space (not normalized)
        - x_min, y_min: top-left corner
        - x_max, y_max: bottom-right corner
    """
    
    def __init__(self, yolo_dataset_path: str, output_path: str, image_size: Tuple[int, int] = (416, 416)):
        """
        Initialize the converter.
        
        Args:
            yolo_dataset_path: Path to YOLO dataset directory (containing train/ and valid/)
            output_path: Path where converted dataset will be saved
            image_size: Target image size (width, height) for pixel coordinate conversion
        """
        self.yolo_dataset_path = Path(yolo_dataset_path)
        self.output_path = Path(output_path)
        self.image_size = image_size
        self.image_width, self.image_height = image_size
        
        # Create output directory if it doesn't exist
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Load class names from data.yaml
        self.class_names = self._load_class_names()
        
    def _load_class_names(self) -> List[str]:
        """
        Load class names from data.yaml file.
        
        Returns:
            List of class names
        """
        yaml_path = self.yolo_dataset_path / "data.yaml"
        
        if not yaml_path.exists():
            print(f"Warning: data.yaml not found at {yaml_path}, using default class '0'")
            return ['0']
        
        # Simple YAML parser (avoiding PyYAML dependency)
        class_names = []
        with open(yaml_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('names:'):
                    continue
                elif line.startswith('- '):
                    class_name = line[2:].strip().strip("'\"")
                    class_names.append(class_name)
        
        if not class_names:
            print("Warning: No class names found in data.yaml, using default class '0'")
            return ['0']
        
        print(f"Loaded {len(class_names)} classes: {class_names}")
        return class_names
    
    def _yolo_to_tf_bbox(self, yolo_bbox: List[float]) -> List[float]:
        """
        Convert a single YOLO bounding box to TensorFlow format.
        
        YOLO format: [class_id, center_x, center_y, width, height] (normalized)
        TensorFlow format: [x_min, y_min, x_max, y_max, class_id] (pixel coordinates)
        
        Args:
            yolo_bbox: YOLO bounding box [class_id, center_x, center_y, width, height]
            
        Returns:
            TensorFlow bounding box [x_min, y_min, x_max, y_max, class_id]
        """
        class_id, center_x, center_y, width, height = yolo_bbox
        
        # Convert normalized coordinates to pixel coordinates
        center_x_px = center_x * self.image_width
        center_y_px = center_y * self.image_height
        width_px = width * self.image_width
        height_px = height * self.image_height
        
        # Calculate corner coordinates
        x_min = center_x_px - (width_px / 2)
        y_min = center_y_px - (height_px / 2)
        x_max = center_x_px + (width_px / 2)
        y_max = center_y_px + (height_px / 2)
        
        # Clip to image boundaries
        x_min = max(0, min(x_min, self.image_width))
        y_min = max(0, min(y_min, self.image_height))
        x_max = max(0, min(x_max, self.image_width))
        y_max = max(0, min(y_max, self.image_height))
        
        return [x_min, y_min, x_max, y_max, class_id]
    
    def _read_yolo_label_file(self, label_path: Path) -> List[List[float]]:
        """
        Read a YOLO label file and convert all bounding boxes.
        
        Args:
            label_path: Path to YOLO label file (.txt)
            
        Returns:
            List of TensorFlow-format bounding boxes
        """
        tf_bboxes = []
        
        if not label_path.exists():
            print(f"Warning: Label file not found: {label_path}")
            return tf_bboxes
        
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Parse YOLO format: class_id center_x center_y width height
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid line in {label_path}: {line}")
                    continue
                
                try:
                    yolo_bbox = [float(part) for part in parts]
                    tf_bbox = self._yolo_to_tf_bbox(yolo_bbox)
                    tf_bboxes.append(tf_bbox)
                except ValueError as e:
                    print(f"Warning: Could not parse line in {label_path}: {line}, error: {e}")
        
        return tf_bboxes
    
    def _find_matching_image(self, label_path: Path, images_dir: Path) -> Optional[Path]:
        """
        Find the image file corresponding to a label file.
        
        Args:
            label_path: Path to label file
            images_dir: Directory containing images
            
        Returns:
            Path to matching image file, or None if not found
        """
        # Get label filename without extension
        label_name = label_path.stem
        
        # Try common image extensions
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
            image_path = images_dir / f"{label_name}{ext}"
            if image_path.exists():
                return image_path
        
        return None
    
    def _process_split(self, split_name: str) -> List[Dict]:
        """
        Process a dataset split (train or valid).
        
        Args:
            split_name: Name of the split ('train' or 'valid')
            
        Returns:
            List of dictionaries containing image and label information
        """
        split_data = []
        
        # Paths for this split
        split_dir = self.yolo_dataset_path / split_name
        labels_dir = split_dir / "labels"
        images_dir = split_dir / "images"
        
        if not labels_dir.exists():
            print(f"Warning: Labels directory not found: {labels_dir}")
            return split_data
        
        if not images_dir.exists():
            print(f"Warning: Images directory not found: {images_dir}")
            return split_data
        
        # Process all label files
        label_files = list(labels_dir.glob("*.txt"))
        print(f"Processing {split_name} split: {len(label_files)} label files")
        
        for label_path in label_files:
            # Find matching image
            image_path = self._find_matching_image(label_path, images_dir)
            
            if image_path is None:
                print(f"Warning: No matching image found for label: {label_path.name}")
                continue
            
            # Read and convert labels
            tf_bboxes = self._read_yolo_label_file(label_path)
            
            # Create data entry
            data_entry = {
                'image_path': str(image_path.relative_to(self.yolo_dataset_path)),
                'label_path': str(label_path.relative_to(self.yolo_dataset_path)),
                'bboxes': tf_bboxes,
                'num_objects': len(tf_bboxes),
                'split': split_name
            }
            
            split_data.append(data_entry)
        
        print(f"Successfully processed {len(split_data)} images in {split_name} split")
        return split_data
    
    def convert_dataset(self) -> Dict:
        """
        Convert the entire YOLO dataset to TensorFlow format.
        
        Returns:
            Dictionary containing converted dataset with 'train' and 'valid' splits
        """
        print("=" * 60)
        print("YOLO to TensorFlow Dataset Converter")
        print("=" * 60)
        print(f"YOLO dataset path: {self.yolo_dataset_path}")
        print(f"Output path: {self.output_path}")
        print(f"Image size: {self.image_size}")
        print(f"Number of classes: {len(self.class_names)}")
        print("=" * 60)
        
        dataset = {
            'metadata': {
                'image_size': self.image_size,
                'num_classes': len(self.class_names),
                'class_names': self.class_names,
                'format': 'tensorflow',
                'bbox_format': 'x_min,y_min,x_max,y_max,class_id'
            },
            'train': [],
            'valid': []
        }
        
        # Process train split
        if (self.yolo_dataset_path / 'train').exists():
            dataset['train'] = self._process_split('train')
        
        # Process valid split
        if (self.yolo_dataset_path / 'valid').exists():
            dataset['valid'] = self._process_split('valid')
        
        # Save dataset to JSON
        output_file = self.output_path / "tensorflow_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(dataset, f, indent=2)
        
        print("=" * 60)
        print("Conversion complete!")
        print(f"Total train samples: {len(dataset['train'])}")
        print(f"Total valid samples: {len(dataset['valid'])}")
        print(f"Dataset saved to: {output_file}")
        print("=" * 60)
        
        return dataset
    
    def create_tf_data_compatible_format(self, dataset: Dict) -> Dict:
        """
        Create a simplified format optimized for tf.data.Dataset.
        
        This creates a structure that can be easily loaded into tf.data:
        - Separate lists for image paths and labels
        - Labels as numpy arrays for efficient loading
        
        Args:
            dataset: Converted dataset dictionary
            
        Returns:
            Dictionary with tf.data-compatible format
        """
        tf_data_format = {
            'metadata': dataset['metadata'],
            'train': {
                'image_paths': [],
                'bboxes': [],
                'num_objects': []
            },
            'valid': {
                'image_paths': [],
                'bboxes': [],
                'num_objects': []
            }
        }
        
        # Process train split
        for entry in dataset['train']:
            tf_data_format['train']['image_paths'].append(entry['image_path'])
            tf_data_format['train']['bboxes'].append(entry['bboxes'])
            tf_data_format['train']['num_objects'].append(entry['num_objects'])
        
        # Process valid split
        for entry in dataset['valid']:
            tf_data_format['valid']['image_paths'].append(entry['image_path'])
            tf_data_format['valid']['bboxes'].append(entry['bboxes'])
            tf_data_format['valid']['num_objects'].append(entry['num_objects'])
        
        # Save tf.data-compatible format
        output_file = self.output_path / "tf_data_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(tf_data_format, f, indent=2)
        
        print(f"tf.data-compatible format saved to: {output_file}")
        
        return tf_data_format


def main():
    """
    Main function to run the YOLO to TensorFlow conversion.
    """
    # Configuration
    YOLO_DATASET_PATH = "LU3M6TGT_yolo_format"
    OUTPUT_PATH = "tensorflow_dataset"
    IMAGE_SIZE = (416, 416)  # Adjust based on your image size
    
    # Create converter
    converter = YOLOToTensorFlowConverter(
        yolo_dataset_path=YOLO_DATASET_PATH,
        output_path=OUTPUT_PATH,
        image_size=IMAGE_SIZE
    )
    
    # Convert dataset
    dataset = converter.convert_dataset()
    
    # Create tf.data-compatible format
    tf_data_format = converter.create_tf_data_compatible_format(dataset)
    
    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    print(f"Train samples: {len(dataset['train'])}")
    print(f"Valid samples: {len(dataset['valid'])}")
    print(f"Total samples: {len(dataset['train']) + len(dataset['valid'])}")
    print(f"Classes: {dataset['metadata']['class_names']}")
    print(f"Image size: {dataset['metadata']['image_size']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
