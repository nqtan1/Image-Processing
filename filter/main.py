from filtre_by_convolution import *

# Import images
img_lena = np.array(Image.open('./img/lena.png'))
img_lena_bruit = np.array(Image.open('./img/lena_bruit.png'))
img_boat = np.array(Image.open('./img/boat.png'))
img_circuit = np.array(Image.open('./img/circuit.png'))

''' Low-pass filtering '''

# 1. Medium Filtering

# Define filters
filters = {
    '3x3 Medium Filter': 1/9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
    '5x5 Medium Filter': 1/25 * np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], 
                                        [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]),
    '3x3 Less Active Filter': 1/10 * np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]]),
    '5x5 Privileged Filter': 1/42 * np.array([[1, 1, 1, 1, 1], [1, 1, 4, 1, 1], [1, 4, 6, 4, 1], 
                                              [1, 1, 4, 1, 1], [1, 1, 1, 1, 1]])
}

# Perform convolution and store results in a dictionary
results = {name: convolve2D(input_matrix=img_lena_bruit, kernel=kernel) for name, kernel in filters.items()}

# Display results using matplotlib
plt.figure(figsize=(12, 8))

# Display the original image and filtered images
images = [img_lena_bruit] + list(results.values())
titles = ['Original Image (Noisy)'] + list(filters.keys())

for i, (image, title) in enumerate(zip(images, titles)):
    plt.subplot(2, 3, i+1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.tight_layout()
plt.show()


# 2. Gaussian filter


kernel_a = gaussian_kernel(sigma=1,size=3)
kernal_b = gaussian_kernel(sigma=1.25,size=5)

result_a = convolve2D(img_lena,kernel=kernel_a)
result_b = convolve2D(img_lena,kernel=kernal_b)

fig, axs = plt.subplots(2,3)

fig.set_size_inches(15, 8, forward=True)

axs[0][0].imshow(img_lena_bruit,cmap="gray")
axs[0][0].set(xlabel='Lena noissy')
axs[0][1].imshow(kernel_a,cmap="gray")
axs[0][1].set(xlabel='Kernel sigma = 1 and kernel size = 3')
axs[0][2].imshow(result_a,cmap="gray")
axs[0][2].set(xlabel='Resultat')

axs[1][0].imshow(img_lena_bruit,cmap="gray")
axs[1][0].set(xlabel='Lena noissy')
axs[1][1].imshow(kernal_b,cmap="gray")
axs[1][1].set(xlabel='Kernel sigma = 1.25 and kernel size = 5')
axs[1][2].imshow(result_b,cmap="gray")
axs[1][2].set(xlabel='Resultat')

plt.show()


''' High-pass filter '''

# 1.Detail Enhancement

F1_kernel = np.array([[0, -1, 0], 
                      [-1, 4, -1], 
                      [0, -1, 0]]) # High-pass filter for detail enhancement

result = convolve2D(img_lena_bruit,kernel=F1_kernel)

fig, axs = plt.subplots(1,3)

fig.set_size_inches(15, 8, forward=True)

axs[0].imshow(img_lena_bruit,cmap="gray")
axs[0].set(xlabel='Lena noissy')
axs[1].imshow(F1_kernel,cmap="gray")
axs[1].set(xlabel='Kernel F1')
axs[2].imshow(result,cmap="gray")
axs[2].set(xlabel='Resultat')

plt.show()


# 2. Détection de structure

# Point kernel

kernel_point = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]])

# Horizontal kernel
kernel_horizontal = np.array([[-1, -1, -1],
                              [ 2,  2,  2],
                              [-1, -1, -1]])

# Vertical kernel
kernel_vertical = np.array([[-1,  2, -1],
                            [-1,  2, -1],
                            [-1,  2, -1]])

# Diagonal kernel
kernel_diagonal = np.array([[-1, -1,  2],
                             [-1,  2, -1],
                             [ 2, -1, -1]])

result_point_lena = convolve2D(img_lena_bruit,kernel=kernel_point)
result_point_boat = convolve2D(img_boat,kernel=kernel_point)
result_horizontal_circuit = convolve2D(img_circuit,kernel=kernel_horizontal)
result_vertical_circuit = convolve2D(img_circuit,kernel=kernel_vertical)

gradient_magnitude = invert_image(np.sqrt(result_horizontal_circuit**2 + result_vertical_circuit**2))

plt.figure(figsize=(10, 5))
plt.title("Gradient Magnitude - Circuit")
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')

plt.show()

fig, axes = plt.subplots(4, 3, figsize=(15, 15))

axes[0, 0].imshow(img_lena_bruit, cmap='gray')
axes[0, 0].set_title("Original Lena (Noise)")
axes[0, 0].axis('off')
axes[0, 1].imshow(kernel_point, cmap='gray')
axes[0, 1].set_title("Point Kernel")
axes[0, 1].axis('off')
axes[0, 2].imshow(result_point_lena, cmap='gray')
axes[0, 2].set_title("Result Lena (Point Kernel)")
axes[0, 2].axis('off')

axes[1, 0].imshow(img_boat, cmap='gray')
axes[1, 0].set_title("Original Boat")
axes[1, 0].axis('off')
axes[1, 1].imshow(kernel_point, cmap='gray')
axes[1, 1].set_title("Point Kernel")
axes[1, 1].axis('off')
axes[1, 2].imshow(result_point_boat, cmap='gray')
axes[1, 2].set_title("Result Boat (Point Kernel)")
axes[1, 2].axis('off')

axes[2, 0].imshow(img_circuit, cmap='gray')
axes[2, 0].set_title("Original Circuit")
axes[2, 0].axis('off')
axes[2, 1].imshow(kernel_horizontal, cmap='gray')
axes[2, 1].set_title("Horizontal Kernel")
axes[2, 1].axis('off')
axes[2, 2].imshow(result_horizontal_circuit, cmap='gray')
axes[2, 2].set_title("Result Circuit (Horizontal Kernel)")
axes[2, 2].axis('off')

axes[3, 0].imshow(img_circuit, cmap='gray')
axes[3, 0].set_title("Original Circuit")
axes[3, 0].axis('off')
axes[3, 1].imshow(kernel_vertical, cmap='gray')
axes[3, 1].set_title("Vertical Kernel")
axes[3, 1].axis('off')
axes[3, 2].imshow(result_vertical_circuit, cmap='gray')
axes[3, 2].set_title("Result Circuit (Vertical Kernel)")
axes[3, 2].axis('off')

plt.tight_layout()
plt.show()


# 3. Détection de contours par filtres de Sobel

# 
kernel_grad_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
kernel_grad_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

result_x = convolve2D(img_circuit,kernel=kernel_grad_x)
result_y = convolve2D(img_circuit,kernel=kernel_grad_y)

# Compute gradient magnitude
gradient_magnitude = invert_image(np.sqrt(result_x**2 + result_y**2))

fig, axs = plt.subplots(1, 4, figsize=(20, 8))

# Original image
axs[0].imshow(img_circuit, cmap="gray")
axs[0].set_title("Original Image")
axs[0].axis("off")

# Grad_x
axs[1].imshow(result_x, cmap="gray")
axs[1].set_title("Gradient X")
axs[1].axis("off")

# Grad_y
axs[2].imshow(result_y, cmap="gray")
axs[2].set_title("Gradient Y")
axs[2].axis("off")

# The resulting gradient magnitude
axs[3].imshow(gradient_magnitude, cmap="gray")
axs[3].set_title("Gradient Magnitude")
axs[3].axis("off")

plt.tight_layout()
plt.show()


'''  Filtrage par algorithme local '''

# 1. Median Filter
kernel_sizes = [3, 5, 7, 9]
median_imgs = [median_filter(input_img=img_lena_bruit, kernel_size=size) for size in kernel_sizes]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image with Noise")
plt.imshow(img_lena_bruit, cmap='gray')
plt.axis('off')

for i, size in enumerate(kernel_sizes):
    plt.subplot(2, 3, i + 2)
    plt.title(f"Kernel Size = {size}")
    plt.imshow(median_imgs[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 2. Conservative Filter 
kernel_sizes = [3, 5, 7, 9, 11]
conservative_imgs = [conservative_filter(input_img=img_lena_bruit, kernel_size=size) for size in kernel_sizes]

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image with Noise")
plt.imshow(img_lena_bruit, cmap='gray')
plt.axis('off')

for i, size in enumerate(kernel_sizes):
    plt.subplot(2, 3, i + 2)
    plt.title(f"Kernel Size = {size}")
    plt.imshow(conservative_imgs[i], cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()