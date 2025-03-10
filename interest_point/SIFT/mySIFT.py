import numpy as np
import cv2
import matplotlib.pyplot as plt
from functools import cmp_to_key
from test_file import *

# Define directory path
dir_path = './img/'

# Function to open an image
def openImage(image_path):
    image = cv2.imread(image_path)
    return image

# Function to display an image
def showImage(image, window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    #cv2.destroyAllWindows()

# Function to display image information
def showImageInformation(image):
    if len(image.shape) == 3:  # Color image (3 channels)
        height, width, channels = image.shape
        print(f'Image dimensions: {width} x {height}, Channels: {channels}')
    elif len(image.shape) == 2:  # Grayscale image (1 channel)
        height, width = image.shape
        print(f'Image dimensions: {width} x {height}, Channels: 1 (Grayscale)')
    else:
        print("Unknown image format")


def convertImageToGray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image

# Image paramid and DoG
def generateBaseImage(image, sigma, assumed_blur):
    '''
    '''
    image = cv2.resize(image, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    sigma_diff = np.sqrt(np.maximum((sigma ** 2) - ((2 * assumed_blur) ** 2), 0.01))
    return cv2.GaussianBlur(image, (0, 0), sigmaX=sigma_diff, sigmaY=sigma_diff)

def computeNumberOfOctaves(image_shape):
    '''
    '''
    return int(np.round(np.log(min(image_shape)) / np.log(2) - 1))

def generateGaussianKernels(sigma, nb_intervals):
    nb_imgs_per_octave = nb_intervals + 3
    k = 2 ** (1. / nb_intervals) # Scale factor
    gauss_kernels = np.zeros(nb_imgs_per_octave)
    gauss_kernels[0] = sigma   

    for index in range(1, nb_imgs_per_octave):
        sigma_prev = (k ** index) * sigma
        sigma_total = k * sigma_prev
        gauss_kernels[index] = np.sqrt(sigma_total ** 2 - sigma_prev ** 2)
    return gauss_kernels

def generateGaussianImages(image, nb_octaves, gaussian_kernels):
    '''
    Generate Gaussian images for each octave.
    '''
    gaussian_images = []

    for octave in range(nb_octaves):  
        gaussian_imgs_in_octave = []
        gaussian_imgs_in_octave.append(image)  # First image in the octave

        # Generate gaussian images in the octave
        for sigma in gaussian_kernels[1:]:  # Correct iteration over the kernels
            image = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
            gaussian_imgs_in_octave.append(image)
        
        # Append the octave to the list of gaussian images
        gaussian_images.append(gaussian_imgs_in_octave)
        
        # Resize the image for the next octave
        octave_base = gaussian_imgs_in_octave[-3]
        image = cv2.resize(octave_base, (octave_base.shape[1] // 2, octave_base.shape[0] // 2), interpolation=cv2.INTER_NEAREST)

    return np.array(gaussian_images, dtype=object)

# Function to generate DoG images
def generateDoGImages(gaussian_images):
    DoG_images = []

    for gaussian_imgs_in_octave in gaussian_images:
        DoG_images_in_octave = []

        for first_img, second_img in zip(gaussian_imgs_in_octave, gaussian_imgs_in_octave[1:]):
            DoG_images_in_octave.append(second_img - first_img)
        DoG_images.append(DoG_images_in_octave)
    return np.array(DoG_images, dtype=object)

# Scale Space Extrema
def isPixelAnExtramum(first_subImg, second_subImg, third_subImg, threshold):
    '''
    Sub Image: 3x3
    '''
    centre_pixel = second_subImg[1, 1]

    if abs(centre_pixel) <= threshold:
        return False

    neighbors = np.concatenate([
        first_subImg.flatten(),
        third_subImg.flatten(),
        second_subImg[0, :],
        second_subImg[2, :],
        [second_subImg[1, 0], second_subImg[1, 2]]
    ])

    if centre_pixel > 0:
        return all(centre_pixel >= neighbors)
    elif centre_pixel < 0:
        return all(centre_pixel <= neighbors)

    return False

def computeGradientAtCenterPixel(pixel_array):
    dx = 0.5 * (pixel_array[1,1,2] - pixel_array[1,1,0])    
    dy = 0.5 * (pixel_array[1,2,1] - pixel_array[1,0,1])
    ds = 0.5 * (pixel_array[2,1,1] - pixel_array[0,1,1])
    return np.array([dx, dy, ds])

def computeHessianAtCenterPixel(pixel_array):   
    center_plixel = pixel_array[1,1,1]  
    
    dxx = pixel_array[1,1,2] - 2 * center_plixel + pixel_array[1,1,0]
    dyy = pixel_array[1,2,1] - 2 * center_plixel + pixel_array[1,0,1]
    dss = pixel_array[2,1,1] - 2 * center_plixel + pixel_array[0,1,1]

    dxy = 0.25 * (pixel_array[1,2,2] - pixel_array[1,0,2] - pixel_array[1,2,0] + pixel_array[1,0,0])
    dxs = 0.25 * (pixel_array[2,1,2] - pixel_array[0,1,2] - pixel_array[2,1,0] + pixel_array[0,1,0])
    dys = 0.25 * (pixel_array[2,2,1] - pixel_array[0,2,1] - pixel_array[2,0,1] + pixel_array[0,0,1])

    return np.array([[dxx, dxy, dxs],    
                     [dxy, dyy, dys],
                     [dxs, dys, dss]])

def localizeExtremum(posX,posY, img_index, octave_index, nb_intervals, nb_DoG_imgs_in_octave, sigma, contrast_threshold, img_border_width,eigenValue_ratio=10, nb_attemps=5):
    extremum_is_outside = False # Flag to check if the extremum is outside the image
    img_shape = nb_DoG_imgs_in_octave[0].shape

    for index in range(nb_attemps):
        first_img, second_img, third_img = nb_DoG_imgs_in_octave[img_index - 1:img_index + 2]
        
        pixel_cube = np.stack([
                                first_img[posX-1:posX+2, posY-1:posY+2],
                                second_img[posX-1:posX+2, posY-1:posY+2],
                                third_img[posX-1:posX+2, posY-1:posY+2]
                            ]).astype('float32') / 255.0
        gradient = computeGradientAtCenterPixel(pixel_cube)
        hessian = computeHessianAtCenterPixel(pixel_cube)
        extremum_update = -np.linalg.lstsq(hessian, gradient, rcond=None)[0] # Solve for the extremum update

        if np.abs(extremum_update[0]) < 0.5 and np.abs(extremum_update[1]) < 0.5 and np.abs(extremum_update[2]) < 0.5:
            break
        
        # Update the position of the extremum
        posY += int(np.round(extremum_update[0]))
        posX += int(np.round(extremum_update[1]))
        
        img_index += int(np.round(extremum_update[2]))

        if posX <img_border_width or posX >= img_shape[0] - img_border_width or posY < img_border_width or posY >= img_shape[1] - img_border_width or img_index < 1 or img_index > nb_intervals:
            extremum_is_outside = True
            break
            
        if extremum_is_outside:
            return None

        if index >= nb_attemps - 1:
            return None
        
        valueAtUpdateExtremum = pixel_cube[1,1,1] + 0.5 * np.dot(gradient, extremum_update)

        # check if the extremum is too low
        if np.abs(valueAtUpdateExtremum) * nb_intervals >= contrast_threshold:
            xy_hessian = hessian[:2, :2]
            xy_hessian_trace = np.trace(xy_hessian)
            xy_hessian_det = np.linalg.det(xy_hessian)

            if xy_hessian_det > 0 and eigenValue_ratio * (xy_hessian_trace ** 2) < ((eigenValue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((posY + extremum_update[0]) * (2 ** octave_index), (posX + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + img_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((img_index + extremum_update[2]) / np.float32(nb_intervals))) * (2 ** (octave_index + 1)) 
                keypoint.response = np.abs(valueAtUpdateExtremum)
                return  keypoint, img_index
        return None              

def findScaleSpaceExtrema(gauss_imgs, DoG_imgs, nb_intervals, sigma, img_border_width, constrast_theshold):
    keypoints = []  
    threshold = np.floor(0.5 * constrast_theshold / nb_intervals)  

    for octave_index, DoG_imgs_in_octave in enumerate(DoG_imgs):
        for img_index, (first_img, second_img, third_img) in enumerate(zip(DoG_imgs_in_octave, DoG_imgs_in_octave[1:], DoG_imgs_in_octave[2:])):
            for posX in range(img_border_width,first_img.shape[0] - img_border_width):
                for posY  in range(img_border_width,first_img.shape[1] - img_border_width):
                    # Check if the pixel is an extremum
                    if isPixelAnExtramum(first_img[posX-1:posX+2,posY-1:posY+2], second_img[posX-1:posX+2,posY-1:posY+2], third_img[posX-1:posX+2,posY-1:posY+2], threshold):
                        loclalization_result = localizeExtremum(posX,posY,img_index+1,octave_index,nb_intervals,DoG_imgs_in_octave,sigma,constrast_theshold,img_border_width)

                        if loclalization_result is not None:
                            keypoint, localized_img_index = loclalization_result
                            keypoints_with_orientations = computeKeypointsWithOrientations(keypoint, octave_index, gauss_imgs[octave_index][localized_img_index])
                            for keypoint_with_orientation in keypoints_with_orientations:
                                    keypoints.append(keypoint_with_orientation)
    return keypoints

# Find orientation of keypoints
def computeKeypointsWithOrientations(keypoint, octave_index, gaussian_img, radius_factor=3, nb_bins=36, peak_ratio=0.8, scale_factor=1.5):
    keypoints_with_orientations = []
    img_shape = gaussian_img.shape

    scale = scale_factor * keypoint.size / np.float32(2 ** (octave_index + 1))  
    radius = int(round(radius_factor*scale))
    weight_factor = -0.5 / (scale ** 2) 
    raw_histogram = np.zeros(nb_bins)
    smooth_histogram = np.zeros(nb_bins)    

    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 **octave_index))) + 1 
        if region_y > 0 and region_y < img_shape[0] - 1:    
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < img_shape[1] - 1:
                    dx = gaussian_img[region_y,region_x+1] - gaussian_img[region_y,region_x-1]
                    dy = gaussian_img[region_y-1,region_x] - gaussian_img[region_y+1,region_x]  
                    grad_magnitude = np.sqrt(dx ** 2 + dy ** 2)
                    grad_orientation = np.rad2deg(np.arctan2(dy, dx)) 
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(np.round(grad_orientation * nb_bins / 360.)) 
                    raw_histogram[histogram_index % nb_bins] += weight * grad_magnitude
    
    for nb in range(nb_bins):
        smooth_histogram[nb] = (6*raw_histogram[nb] +4*(raw_histogram[nb-1] + raw_histogram[(nb+1)%nb_bins]) + raw_histogram[nb-2] + raw_histogram[(nb+2)%nb_bins])/16

    orientation_max = np.max(smooth_histogram) 
    orientation_peaks = np.where(np.logical_and(smooth_histogram > np.roll(smooth_histogram, 1), smooth_histogram > np.roll(smooth_histogram, -1)))[0]

    for peak_index in orientation_peaks:
        peak_value = smooth_histogram[peak_index]
        if peak_value >= peak_ratio * orientation_max:
            left_value = smooth_histogram[(peak_index - 1) % nb_bins]
            right_value = smooth_histogram[(peak_index + 1) % nb_bins]
            interpolated_peak_index = (peak_index + 0.5 * (left_value - right_value) / (left_value - 2 * peak_value + right_value)) % nb_bins
            orientation = 360. - interpolated_peak_index * 360. / nb_bins

            if np.abs(orientation - 360.) < 1e-7:
                orientation = 0
            
            new_keypoint = cv2.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            keypoints_with_orientations.append(new_keypoint)
    
    return keypoints_with_orientations

# Remove double keypoints  

def compareKeypoints(keypoint1, keypoint2):
    if keypoint1.pt[0] != keypoint2.pt[0]:
        return keypoint1.pt[0] - keypoint2.pt[0]
    if keypoint1.pt[1] != keypoint2.pt[1]:
        return keypoint1.pt[1] - keypoint2.pt[1]
    if keypoint1.size != keypoint2.size:
        return keypoint2.size - keypoint1.size
    if keypoint1.angle != keypoint2.angle:
        return keypoint1.angle - keypoint2.angle
    if keypoint1.response != keypoint2.response:
        return keypoint2.response - keypoint1.response
    if keypoint1.octave != keypoint2.octave:
        return keypoint2.octave - keypoint1.octave
    return keypoint2.class_id - keypoint1.class_id

def removeDuplicateKeypoints(keypoints):
    if len(keypoints) < 2:
        return keypoints
    keypoints.sort(key = cmp_to_key(compareKeypoints))
    unique_keypoints = [keypoints[0]]   

    for next_keypoint in keypoints[1:]: 
        last_unique_keypoint = unique_keypoints[-1]
        if last_unique_keypoint.pt[0] != next_keypoint.pt[0] or \
           last_unique_keypoint.pt[1] != next_keypoint.pt[1] or \
           last_unique_keypoint.size != next_keypoint.size or \
           last_unique_keypoint.angle != next_keypoint.angle:
            unique_keypoints.append(next_keypoint)
    return unique_keypoints

# Keypoint scale conversion 
def convertKeypointstoInputImageSize(keypoints):
    converted_keypoints = []
    for keypoint in keypoints:
        keypoint.pt = tuple(0.5 * np.array(keypoint.pt))
        keypoint.size *= 0.5
        keypoint.octave = (keypoint.octave & ~255) | ((keypoint.octave - 1) & 255)
        converted_keypoints.append(keypoint)
    return converted_keypoints

# Descriptor generation

def unpackOctave(keypoint):
    octave = keypoint.octave & 255
    layer = (keypoint.octave >> 8) & 255
    if octave >= 128:
        octave = octave | -128
    scale = 1 / np.float32(1 << octave) if octave >= 0 else np.float32(1 << -octave)
    return octave, layer, scale

def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
    """Generate descriptors for each keypoint"""
    descriptors = []

    for keypoint in keypoints:
        octave, layer, scale = unpackOctave(keypoint)
        gaussian_image = gaussian_images[octave + 1, layer]
        num_rows, num_cols = gaussian_image.shape
        point = np.round(scale * np.array(keypoint.pt)).astype('int')
        bins_per_degree = num_bins / 360.
        angle = 360. - keypoint.angle
        cos_angle = np.cos(np.deg2rad(angle))
        sin_angle = np.sin(np.deg2rad(angle))
        weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
        row_bin_list = []
        col_bin_list = []
        magnitude_list = []
        orientation_bin_list = []
        histogram_tensor = np.zeros((window_width + 2, window_width + 2, num_bins))  # first two dimensions are increased by 2 to account for border effects

        # Descriptor window size (described by half_width) follows OpenCV convention
        hist_width = scale_multiplier * 0.5 * scale * keypoint.size
        half_width = int(np.round(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))  # sqrt(2) corresponds to diagonal length of a pixel
        half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))  # ensure half_width lies within image

        for row in range(-half_width, half_width + 1):
            for col in range(-half_width, half_width + 1):
                row_rot = col * sin_angle + row * cos_angle
                col_rot = col * cos_angle - row * sin_angle
                row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                    window_row = int(np.round(point[1] + row))
                    window_col = int(np.round(point[0] + col))
                    if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                        dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                        dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                        gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                        gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                        weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                        row_bin_list.append(row_bin)
                        col_bin_list.append(col_bin)
                        magnitude_list.append(weight * gradient_magnitude)
                        orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

        for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
            row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
            row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
            if orientation_bin_floor < 0:
                orientation_bin_floor += num_bins
            if orientation_bin_floor >= num_bins:
                orientation_bin_floor -= num_bins

            c1 = magnitude * row_fraction
            c0 = magnitude * (1 - row_fraction)
            c11 = c1 * col_fraction
            c10 = c1 * (1 - col_fraction)
            c01 = c0 * col_fraction
            c00 = c0 * (1 - col_fraction)
            c111 = c11 * orientation_fraction
            c110 = c11 * (1 - orientation_fraction)
            c101 = c10 * orientation_fraction
            c100 = c10 * (1 - orientation_fraction)
            c011 = c01 * orientation_fraction
            c010 = c01 * (1 - orientation_fraction)
            c001 = c00 * orientation_fraction
            c000 = c00 * (1 - orientation_fraction)

            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
            histogram_tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
            histogram_tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

        descriptor_vector = histogram_tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
        # Threshold and normalize descriptor_vector
        threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
        descriptor_vector[descriptor_vector > threshold] = threshold
        descriptor_vector /= max(np.linalg.norm(descriptor_vector), 1e-7)
        # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
        descriptor_vector = np.round(512 * descriptor_vector)
        descriptor_vector[descriptor_vector < 0] = 0
        descriptor_vector[descriptor_vector > 255] = 255
        descriptors.append(descriptor_vector)

    return np.array(descriptors, dtype='float32')


def computeKeypointsAndDescriptors(image, sigma=1.6, num_intervals=3, assumed_blur=0.5, image_border_width=5):
    image = image.astype('float32')
    base_image = generateBaseImage(image, sigma, assumed_blur)
    nb_octaves = computeNumberOfOctaves(base_image.shape)
    gauss_kernels = generateGaussianKernels(sigma, num_intervals)
    gaussian_images = generateGaussianImages(base_image, nb_octaves, gauss_kernels)
    DoG_images = generateDoGImages(gaussian_images)
    keypoints = findScaleSpaceExtrema(gaussian_images, DoG_images, num_intervals, sigma, image_border_width, 0.03)
    keypoints = removeDuplicateKeypoints(keypoints)
    keypoints = convertKeypointstoInputImageSize(keypoints)
    descriptors = generateDescriptors(keypoints, gaussian_images)

    return keypoints, descriptors

'''Function to test'''

def testBaseImageGeneration(image, sigma, assumed_blur):
    print("Testing Base Image Generation...")
    base_image = generateBaseImage(image, sigma, assumed_blur)
    showImage(base_image, 'Base Image')
    showImageInformation(base_image)

def testGaussianKernels(sigma, nb_intervals):
    print("Testing Gaussian Kernels Generation...")
    gauss_kernels = generateGaussianKernels(sigma, nb_intervals)
    print(f"Generated Gaussian Kernels: {gauss_kernels}")

def testGaussianImages(image, nb_octaves, gaussian_kernels):
    print("Testing Gaussian Images Generation...")
    gaussian_images = generateGaussianImages(image, nb_octaves, gaussian_kernels)

    # Display all images using matplotlib
    showImagesMatplotlib(gaussian_images, title="Gaussian Images in Octaves")

def testDoGImages(gaussian_images):
    print("Testing DoG Images Generation...")
    dog_images = generateDoGImages(gaussian_images)
    for octave_idx, octave in enumerate(dog_images):
        print(f"Octave {octave_idx + 1} - DoG Images:")
        for dog_idx, dog in enumerate(octave):
            showImage(dog, f"DoG - Octave {octave_idx + 1} - Image {dog_idx + 1}")

def showImagesMatplotlib(images, title="Octave Images"):
    """
    Display images using matplotlib in a grid format.
    images: List of images (each list of images corresponds to one octave).
    title: Title for the plot.
    """
    num_octaves = len(images)
    
    fig, axes = plt.subplots(num_octaves, len(images[0]), figsize=(12, 12))
    fig.suptitle(title, fontsize=16)

    for octave_idx, octave in enumerate(images):
        for img_idx, img in enumerate(octave):
            ax = axes[octave_idx, img_idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')  # Turn off axis

            ax.set_title(f"Octave {octave_idx + 1} - Image {img_idx + 1}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust to make space for the main title
    plt.show()


def showDoGImagesMatplotlib(DoG_images, title="DoG Images"):
    num_octaves = len(DoG_images)
    num_images_per_octave = len(DoG_images[0])

    fig, axes = plt.subplots(num_octaves, num_images_per_octave, figsize=(15, num_octaves * 3))
    fig.suptitle(title, fontsize=16)

    if num_octaves == 1:
        axes = [axes]  

    for octave_idx, octave in enumerate(DoG_images):
        for img_idx, dog in enumerate(octave):
            ax = axes[octave_idx][img_idx] if num_octaves > 1 else axes[img_idx]
            ax.imshow(dog, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Octave {octave_idx + 1} - DoG {img_idx + 1}")  

    plt.tight_layout(pad=2)
    plt.subplots_adjust(top=0.9, hspace=0.5)
    plt.show()

# Main function
def main():
    MIN_MATCH_COUNT = 10
    print('Welcome to my SIFT Project!')

    # Load and display image
    image_path = dir_path + 'stylo.jpeg'        
    lego_gray = openImage(image_path)  
    lego_gray = convertImageToGray(lego_gray) 
    #showImage(lego_gray, 'Lego Gray')

    # Load and display zoom image
    image_zoom_path = dir_path + 'stylo_zoom.jpeg'
    lego = openImage(image_zoom_path)
    lego = convertImageToGray(lego)
    #showImage(lego, 'Lego')

    # Compute SIFT keypoints and descriptors
    kp1, des1 = computeKeypointsAndDescriptors(lego_gray)
    kp2, des2 = computeKeypointsAndDescriptors(lego)

    # Initialize and use FLANN
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Lowe's ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        # Estimate homography between template and scene
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)[0]

        # Draw detected template in scene image
        h, w = lego_gray.shape
        pts = np.float32([[0, 0],
                          [0, h - 1],
                          [w - 1, h - 1],
                          [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)

        lego = cv2.polylines(lego, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        h1, w1 = lego_gray.shape
        h2, w2 = lego.shape
        nWidth = w1 + w2
        nHeight = max(h1, h2)
        hdif = int((h2 - h1) / 2)
        newimg = np.zeros((nHeight, nWidth, 3), np.uint8)

        for i in range(3):
            newimg[hdif:hdif + h1, :w1, i] = lego_gray
            newimg[:h2, w1:w1 + w2, i] = lego

        # Draw SIFT keypoint matches
        for m in good:
            pt1 = (int(kp1[m.queryIdx].pt[0]), int(kp1[m.queryIdx].pt[1] + hdif))
            pt2 = (int(kp2[m.trainIdx].pt[0] + w1), int(kp2[m.trainIdx].pt[1]))
            cv2.line(newimg, pt1, pt2, (255, 0, 0))

        plt.imshow(newimg)
        plt.show()
    else:
        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))

if __name__ == '__main__':
    main()
