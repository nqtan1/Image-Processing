from PIL import Image
import numpy as np
from scipy.ndimage import laplace,gaussian_filter
import matplotlib.pyplot as plt
from functools import partial

def show_histogram(image):
    '''
    Display the histogram of an input image alongside the original image.

    Input: - image: numpy2D

    Output: - A side-by-side visualization of the histogram and the original image.
    '''
    img_array = np.array(image)
    histogram, bins = np.histogram(img_array, bins=256, range=(0, 255))
    
    plt.figure(figsize=(10, 5))  

    # Histogram 
    plt.subplot(1, 2, 1) 
    plt.bar(bins[:-1], histogram, width=1, color='orange', edgecolor='black')
    plt.title('Histogram of Image')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Original image
    plt.subplot(1, 2, 2) 
    plt.imshow(image, cmap='gray')
    plt.title('Image')
    plt.axis('off') 

    plt.tight_layout()  
    plt.show()


def binarize_image (image_np,threshold):
    '''
    Input:  - image : numpy2D
            - threshold: int 
    
    Convert a grayscale image or a color image into a binary image.

    Output:  - image: numpy2D
    '''
    img_bina = image_np
    for i in range(len(img_bina)):
        for j in range(len(img_bina[i])):
            if img_bina[i][j] > threshold:
                img_bina[i][j] = 255
            elif img_bina[i][j] < threshold:
                img_bina[i][j] = 0
    return img_bina


def otsu_threshold_inter_class_variance(image):
    '''
    Aplly Otsu's methode to find an optimimal threshold value and binarize image 

    Input: - image: numpy2D

    Output: - threshold: the optimimal threshold
            - binary_image: numpy2D
    '''
    if not isinstance(image, np.ndarray):
        image = np.array(image)

    # Calculate the histogram of image 
    histogram, bin_edges = np.histogram(image,bins=256,range=(0,256))

    # Number pixels of image 
    nb_pixels = image.shape[0]*image.shape[1]

    # Init parametres of Otsu's method
    current_value_max = 0 
    threshold = 0
    sum_total = np.sum(np.arange(256)*histogram) # Sum of all intensities
    sum_foreground = 0
    sum_background = 0
    weight_background = 0
    weight_foreground = 0 

    for t in range(256):
        # Calcul weights of foreground and background 
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = nb_pixels - weight_background
        if weight_foreground == 0: 
            break

        # Compute sum itensities of foreground and background
        sum_background  += t*histogram[t]
        sum_foreground  = sum_total - sum_background

        # Compute mean value 
        mean_background = sum_background / weight_background
        mean_foreground = sum_foreground / weight_foreground

        # Compute variance 
        variance = weight_background*weight_foreground*(mean_background-mean_foreground)**2

        if variance > current_value_max: 
            current_value_max = variance
            threshold = t 

    # Binarize image
    binary_image = binarize_image(image_np=image,threshold=threshold)
    return threshold, binary_image

def otsu_threshold_intra_class_variance(image):
    '''
    Apply Otsu's method to find an optimal threshold value based on intra-class variance
    and binarize the image.

    Input: 
    - image: numpy 2D array (grayscale image)

    Output:
    - threshold: The optimal threshold value
    - binary_image: Binarized image (numpy 2D array)
    '''
    if not isinstance(image, np.ndarray):
        image = np.array(image)
        
    # Calculate the histogram of the image
    histogram, bin_edges = np.histogram(image, bins=256, range=(0, 256))

    # Number of pixels in the image
    nb_pixels = image.shape[0] * image.shape[1]

    # Initialize parameters for Otsu's method
    current_min_variance = np.inf
    threshold = 0
    sum_total = np.sum(np.arange(256) * histogram)  # Sum of all intensities
    sum_background = 0
    sum_foreground = 0
    weight_background = 0
    weight_foreground = 0

    # Iterate through all possible threshold values (t)
    for t in range(256):
        # Update weights of background and foreground
        weight_background += histogram[t]
        if weight_background == 0:
            continue

        weight_foreground = nb_pixels - weight_background
        if weight_foreground == 0:
            break

        # Update sum of intensities for background and foreground
        sum_background += t * histogram[t]
        sum_foreground = sum_total - sum_background

        # Calculate mean values for background and foreground
        mean_background = sum_background / weight_background
        mean_foreground = sum_foreground / weight_foreground

        # Compute intra-class variance for background and foreground
        var_background = np.sum(((np.arange(t) - mean_background) ** 2) * histogram[:t]) / weight_background
        var_foreground = np.sum(((np.arange(t+1, 256) - mean_foreground) ** 2) * histogram[t+1:]) / weight_foreground

        # Compute total intra-class variance
        total_intra_class_variance = var_background + var_foreground

        # Update threshold if a lower intra-class variance is found
        if total_intra_class_variance < current_min_variance:
            current_min_variance = total_intra_class_variance
            threshold = t

    # Binarize the image based on the optimal threshold
    binary_image = binarize_image(image, threshold)
    
    return threshold, binary_image


''' CONSTRAST '''

def calculate_contrast(img):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Calculate the standard deviation of the gray levels
    contrast = np.std(img)
    
    return contrast

''' LINEAR '''
def linear_contrast_stretching(img):
    '''
    Input: img: Grayscale image (2D NumPy array).

    Output:  stretched_img: Contrast-stretched image where 
             pixel intensities are rescaled to the range [0, 255].
    '''
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Find the minimum and maximum intensity values in the image
    min_val = np.min(img)
    max_val = np.max(img)
    
    # Apply the linear contrast stretching formula
    stretched_img = ((img - min_val) / (max_val - min_val)) * 255
    stretched_img = stretched_img.astype(np.uint8)
    
    return stretched_img

def linear_contrast_stretching_saturation(img):
    '''
    Input: img: Grayscale image.

    Output: 
    stretched_img: Contrast-stretched image where pixel intensities 
    are rescaled to the range [0, 255],with saturation applied to 
    exclude extreme pixel intensities.
    '''
    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Calculate histogram
    histogram, bin_edges = np.histogram(img, bins=256, range=(0, 255))
    
    # Calculate cumulative sum of the histogram
    cum_sum = np.cumsum(histogram)
    
    # Total number of pixels
    total_pixels = img.size
    
    # Find the min and max intensity values such that 5% of the pixels are below min and 5% are above max
    min_val = np.searchsorted(cum_sum, 0.05 * total_pixels)
    max_val = np.searchsorted(cum_sum, 0.95 * total_pixels)
    
    # Apply the linear contrast stretching formula
    stretched_img = ((img - min_val) / (max_val - min_val)) * 255
    stretched_img = np.clip(stretched_img, 0, 255)  # Ensure values are within [0, 255]
    stretched_img = stretched_img.astype(np.uint8)
    
    return stretched_img

''' NON-LINEAR '''
def nonlinear_contrast_stretching(img, func):
    '''
    Input:
    - img: Grayscale image (2D NumPy array).
    - func: Custom function to apply for contrast stretching.
    
    Outputs:
    - stretched_img: Contrast-stretched image.
    '''

    if not isinstance(img, np.ndarray):
        img = np.array(img)

    # Normalize the image to the range [0,1]
    img_normalized = img/255.0

    # Apply the function 
    stretched_img = func(img_normalized)

    # Scale back to the range [0,255]
    stretched_img = (stretched_img * 255).astype(np.uint8)
    
    return stretched_img

# Square root function
def func_sqrt(img):
    return np.sqrt(img)

# Square function
def func_square(img):
    return np.power(img,2)

def func_gamma_correction(img, gamma=1.0):
    """Applies gamma correction to the image."""
    return np.power(img, gamma)

def func_log_transform(img):
    """Applies logarithmic transformation to the image."""
    # Add a small constant to avoid log(0)
    return np.log1p(img)

def func_histogram_equalization(img):
    """Performs histogram equalization on the image."""
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0, 1])
    cdf = hist.cumsum()  # Cumulative distribution function
    cdf_normalized = cdf / cdf[-1]  # Normalize to range [0,1]
    img_equalized = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    return img_equalized.reshape(img.shape)

def func_sigmoid_transform(img, gain=1.0, cutoff=0.5):
    """Applies sigmoid function to the image."""
    return 1 / (1 + np.exp(-gain * (img - cutoff)))

def func_power_law_transform(img, c=1.0, gamma=1.0):
    """Applies power-law (gamma) transformation."""
    return c * np.power(img, gamma)

def func_gaussian_filter(img, sigma):
    """Applies a Gaussian filter to the image."""
    return gaussian_filter(img, sigma=sigma)

def func_laplacian_filter(img):
    """Applies a Laplacian filter to the image."""
    return laplace(img)


