from PIL import Image
import numpy as np
import matplotlib.pyplot as plt



#print(image_boat_np)

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

    # 
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
    """
    Apply Otsu's method based on minimization of intra-class variance
    
    Args:
    - image: Grayscale image (2D NumPy array).
    
    Returns:
    - threshold: Optimal threshold.
    - binary_image: Binarized image.
    """
    # Calculate histogram (256 gray levels)
    histogram, _ = np.histogram(image, bins=256, range=(0, 255))
    
    # Total number of pixels
    total_pixels = image.shape[0] * image.shape[1]
    
    # Calculate weights (w1, w2), sum and sum of squared intensities
    cum_sum = np.cumsum(histogram)  # w1
    cum_mean = np.cumsum(np.arange(256) * histogram)  # Sum of intensities of the background class
    total_mean = cum_mean[-1]  # Total intensity of the image (weighted sum of foreground + background)
    
    # Avoid division by zero
    cum_sum[cum_sum == 0] = 1
    valid_foreground = total_pixels - cum_sum  # Number of foreground pixels
    valid_foreground[valid_foreground == 0] = 1  # Avoid division by zero
    
    # Calculate within-class variance (\(\sigma_W^2\))
    variance_within_class = (
        (cum_mean / cum_sum) ** 2 * cum_sum +
        ((total_mean - cum_mean) / valid_foreground) ** 2 * valid_foreground
    )
    
    # Select threshold with the smallest within-class variance
    optimal_threshold = np.argmin(variance_within_class)
    
    # Binarize image
    binary_image = np.where(image > optimal_threshold, 255, 0).astype(np.uint8)
    
    return optimal_threshold, binary_image


