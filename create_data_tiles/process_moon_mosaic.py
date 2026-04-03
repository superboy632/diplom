#!/usr/bin/env python3
"""
Script to process LROC WAC Global Morphology Mosaic of the Moon.
Slices the global mosaic into 416x416 pixel tiles with geographic coordinates.
"""

import os
import csv
import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds
from tqdm import tqdm


def pixel_to_geo(transform, pixel_x, pixel_y):
    """
    Convert pixel coordinates to geographic coordinates using the affine transform.
    
    Args:
        transform: Affine transformation matrix from GeoTIFF
        pixel_x: X pixel coordinate
        pixel_y: Y pixel coordinate
    
    Returns:
        tuple: (lon, lat) geographic coordinates
    """
    lon, lat = transform * (pixel_x, pixel_y)
    return lon, lat


def process_geotiff(input_tif, output_dir, window_size=416, stride=208, 
                    coord_precision=6, use_overlap=True, chunk_size=4096):
    """
    Process a GeoTIFF file and generate tiles with geographic coordinates.
    Uses windowed reading to process the image in chunks, reducing memory usage.
    
    Args:
        input_tif: Path to input GeoTIFF file
        output_dir: Directory to save output tiles and CSV
        window_size: Size of the sliding window (default: 416)
        stride: Step size for sliding window (default: 208)
        coord_precision: Number of decimal places for coordinates (default: 6)
        use_overlap: If True, use stride for overlap; if False, use window_size
        chunk_size: Size of chunks to read from the image (default: 4096)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Open the GeoTIFF file
    print(f"Opening GeoTIFF file: {input_tif}")
    with rasterio.open(input_tif) as src:
        # Get image dimensions
        height, width = src.height, src.width
        print(f"Image dimensions: {width}x{height}")
        
        # Get the affine transformation matrix
        transform = src.transform
        print(f"Transform matrix: {transform}")
        
        # Get CRS information
        crs = src.crs
        print(f"CRS: {crs}")
        
        # Handle multi-band images
        if src.count > 1:
            print(f"Multi-band image detected ({src.count} bands). Using first band.")
        
        # Calculate actual stride
        actual_stride = stride if use_overlap else window_size
        print(f"Window size: {window_size}x{window_size}")
        print(f"Stride: {actual_stride}")
        print(f"Chunk size for reading: {chunk_size}x{chunk_size}")
        
        # Prepare CSV file
        csv_path = os.path.join(output_dir, "tiles_metadata.csv")
        csv_data = []
        
        # Process image with sliding window using windowed reading
        print("Processing tiles...")
        total_tiles = 0
        
        # Calculate number of tiles
        num_tiles_x = (width - window_size) // actual_stride + 1
        num_tiles_y = (height - window_size) // actual_stride + 1
        total_estimated = num_tiles_x * num_tiles_y
        
        print(f"Estimated number of tiles: {total_estimated}")
        
        # Iterate over the image with sliding window using windowed reading
        for y in tqdm(range(0, height - window_size + 1, actual_stride), 
                      desc="Processing rows"):
            for x in tqdm(range(0, width - window_size + 1, actual_stride), 
                          desc="Processing columns", leave=False):
                
                # Create a rasterio window for reading this chunk
                # Use windowed reading to load only the required portion
                from rasterio.windows import Window
                read_window = Window(x, y, window_size, window_size)
                
                # Read the window data directly from the file
                window_data = src.read(1, window=read_window)
                
                # Handle edge cases where window might be smaller
                if window_data.shape[0] < window_size or window_data.shape[1] < window_size:
                    # Pad the window if necessary
                    padded = np.zeros((window_size, window_size), dtype=window_data.dtype)
                    padded[:window_data.shape[0], :window_data.shape[1]] = window_data
                    window_data = padded
                
                # Normalize window data to 0-255 range for PNG
                window_min = np.min(window_data)
                window_max = np.max(window_data)
                
                if window_max > window_min:
                    normalized = ((window_data - window_min) / (window_max - window_min) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(window_data, dtype=np.uint8)
                
                # Calculate geographic coordinates for the window corners
                # Top-left corner
                lon_min, lat_max = pixel_to_geo(transform, x, y)
                
                # Bottom-right corner
                lon_max, lat_min = pixel_to_geo(transform, x + window_size, y + window_size)
                
                # Calculate center coordinates
                lat_center = (lat_min + lat_max) / 2
                lon_center = (lon_min + lon_max) / 2
                
                # Format coordinates with specified precision
                lat_min_str = f"{lat_min:.{coord_precision}f}"
                lon_min_str = f"{lon_min:.{coord_precision}f}"
                lat_max_str = f"{lat_max:.{coord_precision}f}"
                lon_max_str = f"{lon_max:.{coord_precision}f}"
                
                # Create filename
                filename = f"{lat_min_str},{lon_min_str},{lat_max_str},{lon_max_str}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Save tile as PNG
                img = Image.fromarray(normalized)
                img.save(filepath)
                
                # Add to CSV data
                csv_data.append({
                    'filename': filename,
                    'lat_min': lat_min_str,
                    'lon_min': lon_min_str,
                    'lat_max': lat_max_str,
                    'lon_max': lon_max_str,
                    'lat_center': f"{lat_center:.{coord_precision}f}",
                    'lon_center': f"{lon_center:.{coord_precision}f}"
                })
                
                total_tiles += 1
        
        # Write CSV file
        print(f"Writing CSV file: {csv_path}")
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['filename', 'lat_min', 'lon_min', 'lat_max', 'lon_max', 
                         'lat_center', 'lon_center']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nProcessing complete!")
        print(f"Total tiles created: {total_tiles}")
        print(f"Tiles saved to: {output_dir}")
        print(f"Metadata saved to: {csv_path}")


def main():
    """Main function to run the tile generation."""
    
    # Configuration
    input_tif = "../Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"
    output_dir = "../dataset_tiles"
    window_size = 416
    stride = 208  # 50% overlap
    coord_precision = 6
    use_overlap = True  # Set to False for no overlap
    
    print("=" * 60)
    print("Moon Mosaic Tile Generator")
    print("=" * 60)
    print(f"Input file: {input_tif}")
    print(f"Output directory: {output_dir}")
    print(f"Window size: {window_size}x{window_size}")
    print(f"Stride: {stride}")
    print(f"Coordinate precision: {coord_precision} decimal places")
    print(f"Overlap: {'Yes' if use_overlap else 'No'}")
    print("=" * 60)
    print()
    
    # Check if input file exists
    if not os.path.exists(input_tif):
        print(f"Error: Input file '{input_tif}' not found!")
        print(f"Please ensure the file is in the current directory.")
        return
    
    # Process the GeoTIFF
    try:
        process_geotiff(
            input_tif=input_tif,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride,
            coord_precision=coord_precision,
            use_overlap=use_overlap
        )
    except Exception as e:
        print(f"Error processing GeoTIFF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
