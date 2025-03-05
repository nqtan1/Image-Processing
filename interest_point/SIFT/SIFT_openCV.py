import cv2
import numpy as np

# Đọc ảnh
img1 = cv2.imread('img/stylo.jpeg', cv2.IMREAD_GRAYSCALE)  # ảnh gốc
img2 = cv2.imread('img/stylo_zoom.jpeg', cv2.IMREAD_GRAYSCALE)  # ảnh thứ hai

# Khởi tạo các phương pháp phát hiện điểm đặc trưng
sift = cv2.SIFT_create()
orb = cv2.ORB_create()
akaze = cv2.AKAZE_create()

# Phát hiện điểm đặc trưng và descriptors cho từng phương pháp
keypoints_sift1, descriptors_sift1 = sift.detectAndCompute(img1, None)
keypoints_sift2, descriptors_sift2 = sift.detectAndCompute(img2, None)

keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(img1, None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(img2, None)

keypoints_akaze1, descriptors_akaze1 = akaze.detectAndCompute(img1, None)
keypoints_akaze2, descriptors_akaze2 = akaze.detectAndCompute(img2, None)

# Phương pháp match: Sử dụng KNN Matcher để so sánh các điểm đặc trưng
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_sift = bf.match(descriptors_sift1, descriptors_sift2)
matches_orb = bf.match(descriptors_orb1, descriptors_orb2)
matches_akaze = bf.match(descriptors_akaze1, descriptors_akaze2)

# Sắp xếp các điểm match theo khoảng cách
matches_sift = sorted(matches_sift, key=lambda x: x.distance)
matches_orb = sorted(matches_orb, key=lambda x: x.distance)
matches_akaze = sorted(matches_akaze, key=lambda x: x.distance)

# Vẽ các kết quả match
img_matches_sift = cv2.drawMatches(img1, keypoints_sift1, img2, keypoints_sift2, matches_sift[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_orb = cv2.drawMatches(img1, keypoints_orb1, img2, keypoints_orb2, matches_orb[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
img_matches_akaze = cv2.drawMatches(img1, keypoints_akaze1, img2, keypoints_akaze2, matches_akaze[:40], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Resize hình ảnh hiển thị (nếu cần thiết)
max_width = 1920  # Chiều rộng tối đa của cửa sổ
max_height = 1080  # Chiều cao tối đa của cửa sổ

# Resize ảnh cho vừa với màn hình
def resize_to_fit_window(img, max_width, max_height):
    height, width = img.shape[:2]
    scaling_factor = min(max_width / width, max_height / height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return cv2.resize(img, (new_width, new_height))

# Resize ảnh match
img_matches_sift_resized = resize_to_fit_window(img_matches_sift, max_width, max_height)
img_matches_orb_resized = resize_to_fit_window(img_matches_orb, max_width, max_height)
img_matches_akaze_resized = resize_to_fit_window(img_matches_akaze, max_width, max_height)

# Tạo cửa sổ và thay đổi kích thước cửa sổ hiển thị
cv2.namedWindow("SIFT Matches", cv2.WINDOW_NORMAL)
cv2.resizeWindow("SIFT Matches", img_matches_sift_resized.shape[1], img_matches_sift_resized.shape[0])

cv2.namedWindow("ORB Matches", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ORB Matches", img_matches_orb_resized.shape[1], img_matches_orb_resized.shape[0])

cv2.namedWindow("AKAZE Matches", cv2.WINDOW_NORMAL)
cv2.resizeWindow("AKAZE Matches", img_matches_akaze_resized.shape[1], img_matches_akaze_resized.shape[0])

# Hiển thị kết quả
cv2.imshow('SIFT Matches', img_matches_sift_resized)
cv2.imshow('ORB Matches', img_matches_orb_resized)
cv2.imshow('AKAZE Matches', img_matches_akaze_resized)

# Đợi người dùng nhấn phím bất kỳ để đóng cửa sổ
cv2.waitKey(0)
cv2.destroyAllWindows()
