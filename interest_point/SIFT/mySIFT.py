import numpy as np
import cv2
import matplotlib.pyplot as plt
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

def computeNumberOfOctaves(image):
    '''
    '''
    return int(np.round(np.log(min(image.shape)) / np.log(2) - 1))

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

def localizeExtremum(posX,posY, img_index, octave_index, nb_intervals, nb_DoG_imgs_in_octave, sigma, contrast_threshold, img_border_width,eigenValue_ratio, nb_attemps):
    extremum_is_outside = False # Flag to check if the extremum is outside the image
    img_shape = nb_DoG_imgs_in_octave[0].shape

    for index in range(nb_attemps):
        first_img, second_img, third_img = nb_DoG_imgs_in_octave[img_index - 1:img_index + 2]
        # 
        pixel_cube = np.stack(first_img[posX-1:posX+2,posY-1:posY+2], second_img[posX-1:posX+2,posY-1:posY+2], third_img[posX-1:posX+2,posY-1:posY+2]).astype('float32')/255.0
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
            xy_hessian_trace = np.trace
            xy_hessian_det = np.linalg.det(xy_hessian)

            if xy_hessian_det > 0 and eigenValue_ratio * (xy_hessian_trace ** 2) < ((eigenValue_ratio + 1) ** 2) * xy_hessian_det:
                keypoint = cv2.KeyPoint()
                keypoint.pt = ((posY + extremum_update[0]) * (2 ** octave_index), (posX + extremum_update[1]) * (2 ** octave_index))
                keypoint.octave = octave_index + img_index * (2 ** 8) + int(round((extremum_update[2] + 0.5) * 255)) * (2 ** 16)
                keypoint.size = sigma * (2 ** ((img_index + extremum_update[2]) / np.float32(nb_intervals))) * (2 ** (octave_index + 1)) 
                keypoint.response = np.abs(valueAtUpdateExtremum)
                return  keypoint, img_index
        return None              



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
    print('Welcome to my SIFT Project!')

    # Load and display image
    image_path = dir_path + 'stylo.jpeg'        
    stylo = openImage(image_path)  
    stylo_gray = convertImageToGray(stylo) 
    #showImage(stylo_gray, 'stylo')

    # Load and display zoom image
    image_zoom_path = dir_path + 'stylo_zoom.jpeg'
    stylo_zoom = openImage(image_zoom_path)
    stylo_zoom_gray = convertImageToGray(stylo_zoom)
    #showImage(stylo_zoom_gray, 'stylo_zoom')

    # Test Gaussian Kernels
    testGaussianKernels(1.6, 3)

    # Generate Gaussian Images and test
    gaussian_kernels = generateGaussianKernels(1.6, 3)  # Assuming you want to use sigma = 1.6 and 3 intervals
    #testGaussianImages(stylo_gray, 3, gaussian_kernels)

    # Generate DoG Images
    gaussian_images = generateGaussianImages(stylo_gray, 3, gaussian_kernels)
    print(gaussian_images.size)
    print(gaussian_images.shape)    

    dog_images = generateDoGImages(gaussian_images)
    print(dog_images.size)
    print(dog_images.shape)
    # Show DoG Images using Matplotlib
    #showDoGImagesMatplotlib(dog_images)

    # Additional Tests or Changes Here (example):
    # Change sigma value and test again
    #testBaseImageGeneration(stylo_gray, 2.0, 1.0)  # Test with new sigma and assumed_blur values
    #testGaussianKernels(2.0, 4)  # Test Gaussian Kernels with new values
    
    #testDoGImages(generateGaussianImages(stylo_gray, 4, generateGaussianKernels(2.0, 4)))  # New test for DoG Images

if __name__ == '__main__':
    main()
