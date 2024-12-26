from traitement_par_histogramme import *

# Import images
image_lena = Image.open('./img/lena.png')
image_boat = Image.open('./img/boat.png')
image_cells = Image.open('./img/cells.png')

image_lena_np = np.array(image_lena)
image_boat_np = np.array(image_boat)
image_cells_np = np.array(image_cells)

''' HISTOGRAM '''
#show_histogram(image=image_lena)
#show_histogram(image=image_boat)

''' BINARIZE '''

# 1. Threshold = constant: 
#lena_bina = binarize_image(image_np=image_lena_np, threshold=120)

#show_histogram(image=lena_bina)

# 2. Otsu's thresholding: 
 
# a. With inter-class variance: 
otsu_thresh_boat, binary_img_boat = otsu_threshold_inter_class_variance(image_boat_np)
#print(otsu_thresh_boat)

otsu_thresh_lena, binary_img_lena = otsu_threshold_inter_class_variance(image_lena_np)
#print(otsu_thresh_lena)

#show_histogram(image=binary_img_boat)
#show_histogram(image=binary_img_lena)

# b. With intra-class variance: 
optimal_threshold, binary_image = otsu_threshold_intra_class_variance(image=image_lena_np)
bina_lena_image_otsu_new = binarize_image(image_np=image_lena_np, threshold=optimal_threshold)
show_histogram(image=bina_lena_image_otsu_new)
