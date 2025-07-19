import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os

def visualize_samples(image_names, comments, num_images=3):
    """
    Display images with their captions using matplotlib
    Better for server environments and notebooks
    """
    
    base_path = "/pscratch/sd/h/haoming/Projects/clip/flickr-dataset/flickr30k_images/flickr30k_images/"
    
    # Create subplots
    fig, axes = plt.subplots(1, min(num_images, len(image_names)), figsize=(15, 5))
    
    # Handle single image case
    if num_images == 1:
        axes = [axes]
    
    for i, (img_name, comment) in enumerate(zip(image_names[:num_images], comments[:num_images])):
        try:
            # Construct full image path
            img_path = os.path.join(base_path, img_name)
            
            if os.path.exists(img_path):
                # Load and display image
                img = Image.open(img_path)
                axes[i].imshow(img)
                axes[i].axis('off')
                
                # Add caption as title (truncate if too long)
                caption = str(comment)
                if len(caption) > 60:
                    caption = caption[:57] + "..."
                axes[i].set_title(caption, fontsize=10, wrap=True, pad=10)
                
                print(f"✓ Displayed: {img_name}")
            else:
                # Show placeholder if image not found
                axes[i].text(0.5, 0.5, f"Image not found:\n{img_name}", 
                           ha='center', va='center', fontsize=8)
                axes[i].set_title(str(comment)[:50], fontsize=10)
                print(f"✗ Not found: {img_path}")
                
        except Exception as e:
            axes[i].text(0.5, 0.5, f"Error loading:\n{img_name}\n{str(e)}", 
                       ha='center', va='center', fontsize=8)
            axes[i].set_title("Error", fontsize=10)
            print(f"✗ Error loading {img_name}: {e}")
    
    plt.tight_layout()
    plt.suptitle("Flickr Dataset Samples", fontsize=14, y=1.02)
    
    # Save the plot instead of showing (better for server environments)
    plt.savefig('flickr_samples.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to 'flickr_samples.png'")
    
    # Optionally show if display is available
    try:
        plt.show()
    except:
        print("Display not available, image saved to file instead")

def load_flickr_data():
    """
    Load specific columns from Flickr CSV using tab-separated parsing.
    
    Tab-separated vs Standard CSV:
    - Standard CSV uses commas (,) as separators  
    - Tab-separated uses tabs (\t) as separators
    - Better for text with commas: "A dog, running in park"
    """
    
    file_path = "/pscratch/sd/h/haoming/Projects/clip/flickr-dataset/flickr30k_images/results.csv"
    
    print(f"Loading Flickr data from: {file_path}")
    print("Using tab-separated parsing...")
    
    try:
        # Load only the specific columns we need
        df = pd.read_csv(file_path, sep='|')
        
        print(f"✓ Success! Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Convert to lists using actual column names
        image_names = df['image_name'].dropna().tolist()
        comments = df[' comment'].dropna().tolist()  # Note the space before 'comment'
        
        print(f"\nExtracted {len(image_names)} image names")
        print(f"Extracted {len(comments)} comments")
        
        # Show samples
        print(f"\nFirst 5 image names:")
        for i, img in enumerate(image_names[:5], 1):
            print(f"  {i}. {img}")
        
        # Visualize first few images with their comments
        print(f"\nVisualizing first 3 images with captions...")
        visualize_samples(image_names[:3], comments[:3])
        
        print(f"\nFirst 3 comments:")
        for i, comment in enumerate(comments[:3], 1):
            # Truncate long comments
            display_comment = str(comment)[:80] + "..." if len(str(comment)) > 80 else str(comment)
            print(f"  {i}. {display_comment}")
    
        
        return image_names, comments
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None

# Main execution
if __name__ == "__main__":
    image_names, comments = load_flickr_data()
    
    if image_names and comments:
        print(f"\n✓ Successfully loaded:")
        print(f"  - {len(image_names)} image names")
        print(f"  - {len(comments)} comments")
    else:
        print("Failed to load data")
