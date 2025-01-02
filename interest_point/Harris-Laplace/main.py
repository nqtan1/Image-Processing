from harris_laplace import *

# Imports image 
img_lego = Image.open('./img/lego.jpg')
img_lego_gray = Image.open('./img/lego_gray.jpg')

img_shape = Image.open('./img/color_shapes.jpg').convert("L")
img_shape_noise = Image.open('./img/color_shapes_noise.jpg').convert("L")     

# Convert to array 
img_lego_np = np.array(img_lego)
img_lego_gray_np = np.array(img_lego_gray)

img_shape_np = np.array(img_shape)
img_shape_noise_np = np.array(img_shape_noise) 

# Change scale 
lego = generate_scaled_images(img_shape_np, 10, 0.7)
#for i, img in enumerate(lego):
#    print(f"Scale {i + 1}: {img.shape}")
'''
display_scaled_images(lego)

gauss_para =generate_scaled_images_with_gaussian(img_shape_np,num_scales=10, scale_factor=0.7,sigma_0=1)
display_scaled_images(gauss_para)
'''
# Detect corners
corners_harris_laplace,scaled_images = harris_laplace(img_shape_np, num_scales=5)
display_all_corners(corners_harris_laplace,scaled_images)
