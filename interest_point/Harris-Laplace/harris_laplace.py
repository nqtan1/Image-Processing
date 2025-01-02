from PIL import Image, ImageFilter
import numpy as np 
import matplotlib.pyplot as plt 
from harris import *

from PIL import Image
import numpy as np

'''GENERATE IMAGES'''
def generate_scaled_images(img_np, num_scales, scale_factor):
    """
    Generate scaled images from the original image.

    Parameters:
        img_np (numpy.ndarray)
        num_scales (int): Number of scaled images to generate.
        scale_factor (float): Scaling factor 

    Returns:
        list: List of scaled images.
    """
    scaled_images = []
    img_resized = img_np.copy()  # Keep the original image
    scaled_images.append(img_resized)  # Add the original image to the list
    
    for _ in range(num_scales):
        # Resize the image using the scale factor
        new_width = int(img_resized.shape[1] * scale_factor)
        new_height = int(img_resized.shape[0] * scale_factor)
        
        # Stop if the image size becomes too small
        if new_width <= 0 or new_height <= 0:
            break
        
        img_resized = np.array(Image.fromarray(img_resized).resize((new_width, new_height)))
        scaled_images.append(img_resized)
    
    return scaled_images

def generate_scaled_images_with_gaussian(img_np, num_scales, scale_factor, sigma_0):
    """
    Generate scaled images and apply Gaussian blur to each scaled image.

    Parameters:
        img_np (numpy.ndarray): Original image as a NumPy array.
        num_scales (int): Number of scaled images to generate.
        scale_factor (float): Scaling factor for resizing (e.g., 0.5 for 50%).
        sigma_0 (float): Initial sigma value for the first scale.

    Returns:
        list: List of scaled and Gaussian-blurred images.
    """
    scaled_images = []
    img_resized = img_np.copy()  # Keep the original image
    scaled_images.append(img_resized)  # Add the original image to the list
    
    for i in range(1, num_scales + 1):
        # Resize the image using the scale factor
        new_width = int(img_resized.shape[1] * scale_factor)
        new_height = int(img_resized.shape[0] * scale_factor)
        
        # Stop if the image size becomes too small
        if new_width <= 0 or new_height <= 0:
            break
        
        img_resized = np.array(Image.fromarray(img_resized).resize((new_width, new_height)))
        
        # Apply Gaussian filter with sigma = n^i * sigma_0
        sigma = (scale_factor ** i) * sigma_0
        ker_size = int(sigma*6+1)
        gauss_ker = gaussian_kernel(ker_size,sigma)
        blurred_img = convolve2D(img_resized, gauss_ker)
        
        # Add the blurred image to the list
        scaled_images.append(blurred_img)
    
    return scaled_images

# Gaussian kernel 
def gaussian_kernel(size:int, sigma:float)-> np.ndarray:
    # Range of values 
    values = np.linspace(-(size//2),size//2,size)
    grid_x, grid_y = np.meshgrid(values, values)

    # Calculate the Gaussian function
    kernel = np.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
    kernel = kernel / (2 * np.pi * sigma**2)

    # Normalize the kernel 
    kernel = kernel/np.sum(kernel)

    return kernel

# Laplacian filter
def laplacian_filter(img):
    laplacian_kernel = np.array([[0, 1, 0],
                                 [1, -4, 1],
                                 [0, 1, 0]])

    # Convolve the image with the Laplacian kernel
    laplacian_img = convolve2D(img, laplacian_kernel)
    return laplacian_img

'''HARRIS-LAPLACE CORNER DETECTION'''
def harris_laplace(img: np.ndarray, num_scales=4, scale_factor=0.7, sigma_0=1, coeff: float = 0.04, threshold: float = 0.1) -> list:
    """
    Perform Harris Corner Detection with Laplacian on a multi-scale pyramid of images.

    Args:
        img (np.ndarray): Input grayscale image.
        num_scales (int): Number of scales to generate for the pyramid.
        scale_factor (float): Scale factor for each subsequent image.
        sigma_0 (float): Base sigma for Gaussian filtering.
        coeff (float): Harris detector constant, typically between 0.04 and 0.06.
        threshold (float): Threshold value for cornerness, to filter out weak corners.
    
    Returns:
        list: List of cornerness maps for each scale.
    """
    # Create Gaussian Pyramid
    scaled_images = generate_scaled_images_with_gaussian(img, num_scales, scale_factor, sigma_0)
    all_corners = []

    # Detect corners at each scale
    for i, scale_img in enumerate(scaled_images):
        print(f"Processing scale {i + 1}...")
        
        # Detect corners using Harris + Laplacian at each scale
        corners = corners_with_harris_laplace(scale_img, coeff, threshold)
        #print(f"Shape of img {i + 1}..."+str(scale_img.shape))
        #print(f"Shape of cor {i + 1}..."+str(corners.shape))
        # Add the result to the list
        all_corners.append(corners)

        # Display the image with detected corners (if needed)
        #display_corners(scale_img, corners)

    return all_corners,scaled_images

def corners_with_harris_laplace(img: np.ndarray, coeff: float = 0.04, threshold: float = 0.1) -> np.ndarray:
    """
    Perform Harris-Laplace Corner Detection on a grayscale image.

    Args:
        img (np.ndarray): Input grayscale image (2D array).
        coeff (float): Harris detector constant, typically between 0.04 and 0.06.
        threshold (float): Threshold value for cornerness, to filter out weak corners.
    
    Returns:
        np.ndarray: Binary cornerness map.
    """

    # Compute the derivatives
    dif_x = convolve2D(img, dx)
    dif_y = convolve2D(img, dy)

    # Compute the second moment matrix
    dif_xx = convolve2D(dif_x, dx)
    dif_yy = convolve2D(dif_y, dy)
    dif_xy = convolve2D(dif_x, dy)

    # Compute the determinant and trace of the second moment matrix
    det = dif_xx * dif_yy - dif_xy**2
    trace = dif_xx + dif_yy

    # Compute the corner response function
    cor = det - coeff * (trace**2)

    cor_laplacian = laplacian_filter(cor)  # Apply Laplacian filter

    # Threshold the corner response to get binary cornerness map
    corness = cor_laplacian > threshold

    return corness

''' DISPLAY FUNCTIONS'''
def display_scaled_images(scaled_images):
    fig, axes = plt.subplots(1, len(scaled_images), figsize=(15, 5))
    
    for ax, img in zip(axes, scaled_images):
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        
        # Get image size (height and width)
        height, width = img.shape[:2]
        
        # Display the image size on the axis
        ax.set_title(f"{width} x {height}")
        
    plt.show()

def display_corners(img, corners):
    plt.imshow(img, cmap='gray')
    plt.scatter(np.where(corners)[1], np.where(corners)[0], color='red', s=5)
    plt.axis('off')
    plt.show()

def display_all_corners(corners_list, scaled_images):
    """
    Display all images with detected corners in a grid.

    Args:
        corners_list (list): List of binary cornerness maps.
        scaled_images (list): List of scaled images corresponding to corners_list.
    """
    num_scales = len(scaled_images)
    print(num_scales)
    cols = 3  # Number of columns in the grid
    rows = (num_scales + cols - 1) // cols  # Calculate number of rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    fig.suptitle("Harris Laplace with Multiple Scales", fontsize=16, fontweight='bold', y=0.98) 

    # Flatten the axes array for easier iteration
    axes = axes.ravel() if rows > 1 else [axes]

    for i, ax in enumerate(axes):
        if i < num_scales:
            # Plot the image with corners
            ax.imshow(scaled_images[i], cmap='gray')
            corners = corners_list[i]
            ax.scatter(np.where(corners)[1], np.where(corners)[0], color='red', s=5)  # Mark corners
            ax.set_title(f"Scale {i + 1}")
            ax.axis('off')
        else:
            # Hide extra subplots if any
            ax.axis('off')

    plt.tight_layout()
    plt.show()