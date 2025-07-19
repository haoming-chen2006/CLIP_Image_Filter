import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def explore_flickr_dataset():
    """Explore the Flickr dataset CSV and visualize some samples"""
    
    file_path = "/pscratch/sd/h/haoming/Projects/clip/flickr-dataset/flickr30k_images/results.csv"
    
    try:
        # Try different CSV reading approaches to handle parsing issues
        print(f"Attempting to load: {file_path}")
        
        # First, let's examine the file structure
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:10]
            print("First 10 lines of the file:")
            for i, line in enumerate(lines):
                print(f"Line {i+1}: {repr(line[:100])}")
        
        print("\nTrying different CSV parsing methods...")
        
        # Method 1: Standard read_csv
        try:
            df = pd.read_csv(file_path)
            print("✓ Standard CSV parsing successful")
        except:
            # Method 2: Try with different separator
            try:
                df = pd.read_csv(file_path, sep='\t')
                print("✓ Tab-separated parsing successful")
            except:
                # Method 3: Try with error handling
                try:
                    df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
                    print("✓ CSV parsing with error handling successful")
                except:
                    # Method 4: Try with quoting
                    try:
                        df = pd.read_csv(file_path, quoting=3)  # QUOTE_NONE
                        print("✓ CSV parsing without quoting successful")
                    except:
                        # Method 5: Manual parsing
                        print("Standard methods failed, trying manual parsing...")
                        df = pd.read_csv(file_path, sep=',', quotechar='"', skipinitialspace=True, on_bad_lines='skip')
                        print("✓ Manual CSV parsing successful")
        
        print("=" * 60)
        print("FLICKR DATASET EXPLORATION")
        print("=" * 60)
        
        # Basic dataset info
        print(f"Dataset shape: {df.shape}")
        print(f"Number of images: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print()
        
        # Show column names
        print("Column names:")
        for i, col in enumerate(df.columns):
            print(f"  {i+1}. {col}")
        print()
        
        # Show first few rows
        print("First 5 rows:")
        print(df.head())
        print()
        
        # Show data types
        print("Data types:")
        print(df.dtypes)
        print()
        
        # Check for missing values
        print("Missing values per column:")
        print(df.isnull().sum())
        print()
        
        # If there's a caption column, show some examples
        caption_cols = [col for col in df.columns if 'caption' in col.lower() or 'comment' in col.lower() or 'description' in col.lower()]
        if caption_cols:
            print(f"Found caption column(s): {caption_cols}")
            for col in caption_cols[:1]:  # Show first caption column
                print(f"\nSample captions from '{col}':")
                sample_captions = df[col].dropna().head(10)
                for i, caption in enumerate(sample_captions, 1):
                    print(f"  {i}. {caption}")
        
        # If there's an image filename column, show some examples
        image_cols = [col for col in df.columns if 'image' in col.lower() or 'filename' in col.lower() or 'file' in col.lower()]
        if image_cols:
            print(f"\nFound image column(s): {image_cols}")
            for col in image_cols[:1]:  # Show first image column
                print(f"\nSample image filenames from '{col}':")
                sample_images = df[col].dropna().head(5)
                for i, img in enumerate(sample_images, 1):
                    print(f"  {i}. {img}")
        
        # Show unique values for categorical columns
        for col in df.columns:
            if df[col].dtype == 'object' and df[col].nunique() < 20:
                print(f"\nUnique values in '{col}' ({df[col].nunique()} unique):")
                print(df[col].value_counts().head(10))
        
        print("\n" + "=" * 60)
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Please make sure the Flickr dataset has been downloaded.")
        return None
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def visualize_samples(df, num_samples=6):
    """Visualize some image-caption pairs"""
    
    if df is None:
        print("No dataset to visualize")
        return
    
    # Try to find image and caption columns
    image_cols = [col for col in df.columns if 'image' in col.lower() or 'filename' in col.lower()]
    caption_cols = [col for col in df.columns if 'caption' in col.lower() or 'comment' in col.lower() or 'description' in col.lower()]
    
    if not image_cols or not caption_cols:
        print("Could not find image or caption columns for visualization")
        return
    
    image_col = image_cols[0]
    caption_col = caption_cols[0]
    
    print(f"\nAttempting to visualize using:")
    print(f"  Image column: {image_col}")
    print(f"  Caption column: {caption_col}")
    
    # Get base directory for images
    base_dir = "/pscratch/sd/h/haoming/Projects/clip/flickr-dataset/flickr30k_images/"
    
    # Sample some rows
    sample_df = df.dropna(subset=[image_col, caption_col]).sample(min(num_samples, len(df)))
    
    # Create subplots
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(sample_df.iterrows()):
        if i >= num_samples:
            break
            
        try:
            # Try to load the image
            img_path = os.path.join(base_dir, row[image_col])
            if not os.path.exists(img_path):
                # Try alternative paths
                alt_paths = [
                    os.path.join(base_dir, "images", row[image_col]),
                    os.path.join(base_dir, "flickr30k_images", row[image_col]),
                    row[image_col] if os.path.exists(row[image_col]) else None
                ]
                for alt_path in alt_paths:
                    if alt_path and os.path.exists(alt_path):
                        img_path = alt_path
                        break
                else:
                    print(f"Could not find image: {row[image_col]}")
                    continue
            
            # Load and display image
            img = Image.open(img_path)
            axes[i].imshow(img)
            axes[i].axis('off')
            
            # Add caption as title (truncate if too long)
            caption = str(row[caption_col])
            if len(caption) > 50:
                caption = caption[:47] + "..."
            axes[i].set_title(caption, fontsize=10, wrap=True)
            
        except Exception as e:
            print(f"Error loading image {row[image_col]}: {e}")
            axes[i].text(0.5, 0.5, f"Image not found:\n{row[image_col]}", 
                        ha='center', va='center', transform=axes[i].transAxes)
            axes[i].set_title(str(row[caption_col])[:50], fontsize=10)
    
    # Hide unused subplots
    for i in range(len(sample_df), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle("Flickr Dataset Samples: Images and Captions", fontsize=16, y=1.02)
    plt.show()

if __name__ == "__main__":
    # Explore the dataset
    df = explore_flickr_dataset()
    
    # Visualize some samples
    if df is not None:
        print("\nWould you like to visualize some image-caption pairs? (y/n)")
        response = input().lower().strip()
        if response == 'y' or response == 'yes':
            visualize_samples(df)