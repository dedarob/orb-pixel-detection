import cv2
import matplotlib.pyplot as plt


img1 = cv2.imread('pecas.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('cavalo.png', cv2.IMREAD_GRAYSCALE)


orb = cv2.ORB_create()


kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)


bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)


img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)


matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.imshow(img1_kp, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img2_kp, cmap='gray')
plt.axis('off')

plt.subplot(2, 1, 2)  

plt.imshow(matched_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()