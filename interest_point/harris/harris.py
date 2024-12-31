import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 

# Masks equivalent to finite differences
dx = np.array([[1/2,0,-1/2]])
dy = np.array([[1/2,0,-1/2]]).reshape(-1,1)
dxx = np.array([[1,-2,1]])
dyy = np.array([[1,-2,1]]).reshape(-1,1)
dxy = np.array([[1/4,0,-1/4],
               [0,0,0],
               [-1/4,0,1/4]])
delta = np.array([[0,1,0],
                  [1,-4,1],
                  [0,1,0]])

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

# Harris Corner Detection
def harris_corners(img: np.ndarray, coeff: float = 0.04, threshold: float = 0.1) -> np.ndarray:
    """
    Perform Harris Corner Detection on a grayscale image.

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

    # Threshold the corner response to get binary cornerness map
    corness = cor > threshold

    return corness

def harris_corners_color(img: np.ndarray, coeff: float = 0.04, threshold: float = 0.00001) -> np.ndarray:
    """
    Perform Harris Corner Detection on a color image by applying the detection on each channel.
    
    Args:
        img (np.ndarray): Input color image (3D array with shape HxWx3).
        coeff (float): Harris detector constant.
        threshold (float): Threshold to determine corners.
    
    Returns:
        np.ndarray: Binary cornerness map (same size as input image).
    """
    # Split the image into three channels (Red, Green, Blue)
    img_r = img[:, :, 0]  # Red channel
    img_g = img[:, :, 1]  # Green channel
    img_b = img[:, :, 2]  # Blue channel
    
    # Apply Harris Corner Detection for each channel
    cor_r = harris_corners(img_r, coeff, threshold)
    cor_g = harris_corners(img_g, coeff, threshold)
    cor_b = harris_corners(img_b, coeff, threshold)
    
    cor_combined = np.stack([cor_r, cor_g, cor_b], axis=-1)
    cor_total = cor_r | cor_g | cor_b  # Avoid repetition across multiple channels
    
    return cor_combined, cor_total