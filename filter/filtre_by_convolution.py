import numpy as np 
from PIL import Image
import random
import matplotlib.pyplot as plt 

def invert_image(image: np.ndarray) -> np.ndarray:
    """Invert the image by subtracting pixel values from 255"""
    return 255 - image

# Convolution without padding and stride
def convolve2D_no_padding(matrix: np.ndarray , kernel: np.ndarray) -> np.ndarray:
    # Dimension of matrix and kernel
    matrix_h, matrix_w = matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Dinmension of output matrix 
    output_matrix_h = matrix_h - kernel_h + 1 
    output_matrix_w = matrix_w - kernel_w + 1 

    # Initialize the output matrix
    output_matrix = np.zeros((output_matrix_h,output_matrix_w))

    # Compute
    for row in range (output_matrix_h):
        for col in range(output_matrix_w):
            window = matrix[row:row+kernel_h, col:col+kernel_w]
            output_matrix[row,col] = np.sum(window*kernel)

    return output_matrix

# Convolution with auto-padding (same) and stride
def convolve2D(input_matrix: np.ndarray, kernel: np.ndarray, stride: int = 1) -> np.ndarray:
    """
    Perform 2D convolution with automatic padding and stride.

    Args:
        input_matrix (np.ndarray): Input matrix (2D).
        kernel (np.ndarray): Kernel for the convolution (2D).
        stride (int): Stride for the convolution.

    Returns:
        np.ndarray: Output matrix after convolution.
    """
    # Dimension of input matrix and kernel 
    input_matrix_h, input_matrix_w = input_matrix.shape
    kernel_h, kernel_w = kernel.shape

    # Compute automatic padding 
    padding_h = (kernel_h - 1) // 2
    padding_w = (kernel_w - 1) // 2

    # Add zero padding 
    padded_matrix = np.pad(input_matrix, 
                           ((padding_h, padding_h), (padding_w, padding_w)), 
                           mode='constant', constant_values=0)    
    #print(f"Matrix after padding:\n{padded_matrix}\n")

    # Dimension of the padded matrix 
    padded_h, padded_w = padded_matrix.shape

    # Dimension of the output matrix 
    output_matrix_h = (padded_h - kernel_h) // stride + 1
    output_matrix_w = (padded_w - kernel_w) // stride + 1

    # Initialize the output matrix 
    output_matrix = np.zeros((output_matrix_h, output_matrix_w))

    # Compute convolution
    for row in range(output_matrix_h):
        for col in range(output_matrix_w):
            # Compute the start and end indices for the current sliding window
            start_row = row * stride
            start_col = col * stride
            end_row = start_row + kernel_h
            end_col = start_col + kernel_w

            # Extract the sliding window and compute convolution
            window = padded_matrix[start_row:end_row, start_col:end_col]
            output_matrix[row, col] = np.sum(window * kernel)

    return output_matrix

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

# Quick Sort 
def quickSort(arr: list[float]) -> list[float]:
    if len(arr) <= 1:
        return arr
    
    pivot = random.choice(arr)
    left_side = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right_side = [x for x in arr if x > pivot]

    return quickSort(left_side) + middle + quickSort(right_side)
 
# Median Filter
def median_filter(input_img: np.ndarray, kernel_size: int) -> np.ndarray:
    # Dimension of input image and kernel 
    input_img_h, input_img_w = input_img.shape

    half_kernel_size = kernel_size//2

    # Dimension of output image 
    output_img = np.zeros_like(input_img)

    for row in range(half_kernel_size, input_img_h - half_kernel_size): 
        for col in range(half_kernel_size, input_img_w - half_kernel_size): 
            window = input_img[row - half_kernel_size:row + half_kernel_size + 1, 
                               col - half_kernel_size:col + half_kernel_size + 1]
            
            median_value = np.median(window)
            
            # Set the corresponding pixel in the output image
            output_img[row, col] = median_value

    return output_img

# Conservative Filter 
def conservative_filter(input_img:np.ndarray, kernel_size:int)->np.ndarray:
    # Dimension of filter and input image
    half_kernel_size = kernel_size//2 
    input_img_h,input_img_w = input_img.shape

    # Dimension of output image
    output_image = input_img.copy()

    for row in range(half_kernel_size,input_img_h-half_kernel_size):
        for col in range(half_kernel_size,input_img_w-half_kernel_size):
            # Get window value 
            window = input_img[row - half_kernel_size:row + half_kernel_size + 1, 
                               col - half_kernel_size:col + half_kernel_size + 1]

            # Get the neighboor value
            flattened_window = window.flatten()
            #print(flattened_window)
            center_pixel = input_img[row, col]
            flattened_window = np.delete(flattened_window, len(flattened_window) // 2)
            #print(flattened_window)

            # Get min and max value of the window 
            max_val = np.max(window)
            min_val = np.min(window)

            # Check the condition 
            if center_pixel > max_val:
                output_image[row,col] = max_val
            elif center_pixel < min_val:
                output_image[row,col] = min_val
            else:
                output_image[row,col] = center_pixel

    return output_image

