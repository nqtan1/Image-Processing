from harris import *

# Import images 
img_chess = Image.open('./img/chess_board.png').convert("L")
img_chess_np = np.array(img_chess)
img_chess_noise = Image.open('./img/chess_board_noise.png').convert("L")
img_chess_noise_np = np.array(img_chess_noise)

img_shape = Image.open('./img/color_shapes.jpg')
img_shape_noise = Image.open('./img/color_shapes_noise.jpg')
img_shape_np = np.array(img_shape)
img_shape_noise_np = np.array(img_shape_noise)

''' GRAY IMAGE '''
cor = harris_corners(img=img_chess_np)
cor_noise = harris_corners(img=img_chess_noise_np)


fig, axes = plt.subplots(2, 2, figsize=(12, 12))

# Display original image
axes[0, 0].imshow(img_chess_np, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

# Display original image with corner points
axes[0, 1].imshow(img_chess_np, cmap='gray')
axes[0, 1].set_title('Original Image + Harris Corners')
axes[0, 1].axis('off')
axes[0, 1].scatter(np.where(cor)[1], np.where(cor)[0], color='red', s=10, marker='x')  # Plot corner points

# Display image with noise
axes[1, 0].imshow(img_chess_noise_np, cmap='gray')
axes[1, 0].set_title('Image with Noise')
axes[1, 0].axis('off')

# Display image with noise and corner points
axes[1, 1].imshow(img_chess_noise_np, cmap='gray')
axes[1, 1].set_title('Image with Noise + Harris Corners')
axes[1, 1].axis('off')
axes[1, 1].scatter(np.where(cor_noise)[1], np.where(cor_noise)[0], color='red', s=10, marker='x')  # Plot corner points

plt.tight_layout()
plt.show()

''' COLOR IMAGE '''
# Apply Harris Corner Detection to the image
cor_combined, cor_total = harris_corners_color(img_shape_np)
cor_combined_noise, cor_total_noise = harris_corners_color(img_shape_noise_np)

# Display the original image and the cornerness results for each channel, along with the combined channel
fig, axes = plt.subplots(2, 5, figsize=(20, 12))  # Added an extra row for the noisy image

# Original image
axes[0, 0].imshow(img_shape_np)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

# Harris Corner Detection results for each channel (original image)
axes[0, 1].imshow(cor_combined[:, :, 0], cmap='gray')
axes[0, 1].set_title("Red Channel Corners")
axes[0, 1].axis("off")

axes[0, 2].imshow(cor_combined[:, :, 1], cmap='gray')
axes[0, 2].set_title("Green Channel Corners")
axes[0, 2].axis("off")

axes[0, 3].imshow(cor_combined[:, :, 2], cmap='gray')
axes[0, 3].set_title("Blue Channel Corners")
axes[0, 3].axis("off")

# Combined channel for the three channels (original image)
axes[0, 4].imshow(cor_total, cmap='gray')
axes[0, 4].set_title("Combined Corners")
axes[0, 4].axis("off")

# Display noisy image
axes[1, 0].imshow(img_shape_noise_np)
axes[1, 0].set_title("Noisy Image")
axes[1, 0].axis("off")

# Harris Corner Detection results for each channel (noisy image)
axes[1, 1].imshow(cor_combined_noise[:, :, 0], cmap='gray')
axes[1, 1].set_title("Red Channel Corners (Noise)")
axes[1, 1].axis("off")

axes[1, 2].imshow(cor_combined_noise[:, :, 1], cmap='gray')
axes[1, 2].set_title("Green Channel Corners (Noise)")
axes[1, 2].axis("off")

axes[1, 3].imshow(cor_combined_noise[:, :, 2], cmap='gray')
axes[1, 3].set_title("Blue Channel Corners (Noise)")
axes[1, 3].axis("off")

# Combined channel for the three channels (noisy image)
axes[1, 4].imshow(cor_total_noise, cmap='gray')
axes[1, 4].set_title("Combined Corners (Noise)")
axes[1, 4].axis("off")

# Display results
plt.tight_layout()
plt.show()
