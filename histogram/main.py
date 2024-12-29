from traitement_par_histogramme import *

# Import images
image_lena = Image.open('./img/lena.png')
image_boat = Image.open('./img/boat.png')
image_cells = Image.open('./img/cells.png')
image_blur = Image.open('./img/blurry_image.jpeg')
image_calthedrale = Image.open('./img/cathedrale.png')
image_sombre = Image.open('./img/sombre.png')

image_lena_np = np.array(image_lena)
image_boat_np = np.array(image_boat)
image_cells_np = np.array(image_cells)
image_blur_np = np.array(image_blur)
image_cathedrale_np = np.array(image_calthedrale)
image_sombre_np = np.array(image_sombre)

''' HISTOGRAM '''
#show_histogram(image=image_lena)
#show_histogram(image=image_boat)

''' BINARIZE '''

# 1. Threshold = constant: 
#lena_bina = binarize_image(image_np=image_lena_np, threshold=120)
#show_histogram(image=lena_bina)

# 2. Otsu's thresholding: 
 
# a. With inter-class variance: 
#otsu_thresh_boat, binary_img_boat = otsu_threshold_inter_class_variance(image_boat_np)
#print(otsu_thresh_boat)

#otsu_thresh_lena, binary_img_lena = otsu_threshold_inter_class_variance(image_lena_np)
#print(otsu_thresh_lena)

#show_histogram(image=binary_img_boat)
#show_histogram(image=binary_img_lena)

# b. With intra-class variance: 
#otsu_thresh_boat, binary_img_boat = otsu_threshold_intra_class_variance(image_boat)
#print(otsu_thresh_boat)
#show_histogram(image=binary_img_boat)


''' CONTRAST '''

'''1 - LINEAR '''
#contrast_value = calculate_contrast(img=image_blur)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 14.006

#contrast_value = calculate_contrast(img=image_cells)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 33.711
#show_histogram(image=image_cells)

# Apply linear contrast stretching
#stretched_img = linear_contrast_stretching(image_blur)
#contrast_value = calculate_contrast(img=stretched_img)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 37.213
#show_histogram(image=stretched_img)

#stretched_img = linear_contrast_stretching(image_cells)
#contrast_value = calculate_contrast(img=stretched_img)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 43.619
#show_histogram(image=stretched_img)

# Apply linear contrast stretching saturation
#stretched_saturation_img = linear_contrast_stretching_saturation(image_blur)
#contrast_value = calculate_contrast(img=stretched_img)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 70.361
#show_histogram(image=stretched_saturation_img)

#stretched_saturation_img = linear_contrast_stretching_saturation(image_cells)
#contrast_value = calculate_contrast(img=stretched_img)
#print(f'The contrast value of the image is: {contrast_value:.3f}') # = 95.698
#show_histogram(image=stretched_saturation_img)

'''2. NON-LINEAR '''
'''
# Define all functions and prepare their variations with parameters
transform_functions = {
    'Original Image': None,  # Placeholder for the original image
    'Square Root': func_sqrt,
    'Square': func_square,
    'Gamma Correction (gamma=5.0)': partial(func_gamma_correction, gamma=5.0),
    'Log Transform': func_log_transform,
    'Histogram Equalization': func_histogram_equalization,
    'Sigmoid (gain=10, cutoff=0.5)': partial(func_sigmoid_transform, gain=10, cutoff=0.5),
    'Power-Law (c=1.0, gamma=2.0)': partial(func_power_law_transform, c=1.0, gamma=2.0),
    'Gaussian Filter (sigma=3.0)': partial(func_gaussian_filter, sigma=3.0),
    'Laplacian Filter': func_laplacian_filter
}

# Apply all transformations
results = {'Original Image': image_cathedrale_np}
for name, func in transform_functions.items():
    if func is not None:
        results[name] = nonlinear_contrast_stretching(image_cathedrale_np, func)

# Plot all results
num_results = len(results)
cols = 4  # Number of columns for the subplot grid
rows = (num_results + cols - 1) // cols  # Calculate required rows

plt.figure(figsize=(18, 4 * rows))  # Adjust figsize for smaller images
for i, (name, result) in enumerate(results.items(), start=1):
    plt.subplot(rows, cols, i)
    plt.imshow(result, cmap='gray')
    plt.title(name, fontsize=10)  # Adjust font size for better fit
    plt.axis('off')

# Add spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.3)  # Add space between rows and columns
plt.tight_layout()
plt.show()
'''