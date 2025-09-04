#!/usr/bin/env python3

from PIL import Image
import numpy as np
import os

def remove_checkerboard_background(image_path, output_path):
    """Remove checkerboard transparency background from PNG image"""
    
    # Load image
    img = Image.open(image_path).convert('RGBA')
    data = np.array(img)
    
    print(f"Image shape: {data.shape}")
    
    # Sample some background pixels to find actual checkerboard colors
    # Check corners and edges for background colors
    corner_colors = []
    h, w = data.shape[:2]
    
    # Sample corner and edge pixels
    sample_points = [
        (0, 0), (0, w-1), (h-1, 0), (h-1, w-1),  # corners
        (10, 10), (10, w-10), (h-10, 10), (h-10, w-10)  # near corners
    ]
    
    for y, x in sample_points:
        if 0 <= y < h and 0 <= x < w:
            corner_colors.append(tuple(data[y, x, :3]))
    
    # Find the two most common background colors
    from collections import Counter
    color_counts = Counter(corner_colors)
    common_colors = color_counts.most_common(2)
    
    print(f"Most common background colors: {common_colors}")
    
    if len(common_colors) >= 2:
        color1 = np.array(common_colors[0][0])
        color2 = np.array(common_colors[1][0])
        
        # Create mask for these background colors with some tolerance
        tolerance = 5
        mask1 = np.all(np.abs(data[:, :, :3] - color1) <= tolerance, axis=2)
        mask2 = np.all(np.abs(data[:, :, :3] - color2) <= tolerance, axis=2)
        background_mask = mask1 | mask2
        
        # Make background areas fully transparent
        data[background_mask] = [0, 0, 0, 0]  # Fully transparent
        
        print(f"Removed {np.sum(background_mask)} background pixels")
    
    # Create new image with cleaned transparency
    cleaned_img = Image.fromarray(data)
    cleaned_img.save(output_path, 'PNG')
    print(f"✅ Cleaned {image_path} -> {output_path}")

def main():
    # Clean both images
    os.makedirs('images/cleaned', exist_ok=True)
    
    remove_checkerboard_background('images/jaffle.png', 'images/cleaned/jaffle.png')
    remove_checkerboard_background('images/flower.png', 'images/cleaned/flower.png')
    
    print("🎉 All images cleaned! Check images/cleaned/ folder")

if __name__ == "__main__":
    main()