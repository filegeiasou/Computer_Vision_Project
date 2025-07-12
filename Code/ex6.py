import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pathlib import Path

# Load images in grayscale
image1_path = Path("im1.jpg")
image2_path = Path("im2.jpg")

img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)

# Helper function to compute histogram
def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Helper function to apply manual histogram equalization
def manual_hist_equalization(image):
    hist = compute_histogram(image)
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_img = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)
    return equalized_img

# Helper function to compute DCT2
def compute_dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Apply histogram equalization manually and with OpenCV
img1_manual_eq = manual_hist_equalization(img1)
img2_manual_eq = manual_hist_equalization(img2)
img1_cv_eq = cv2.equalizeHist(img1)
img2_cv_eq = cv2.equalizeHist(img2)

# Compute histograms
histograms = {
    "img1_original": compute_histogram(img1),
    "img1_manual_eq": compute_histogram(img1_manual_eq),
    "img1_cv_eq": compute_histogram(img1_cv_eq),
    "img2_original": compute_histogram(img2),
    "img2_manual_eq": compute_histogram(img2_manual_eq),
    "img2_cv_eq": compute_histogram(img2_cv_eq)
}

# Compute DCT2
dct2_images = {
    "img1_original": compute_dct2(img1),
    "img1_manual_eq": compute_dct2(img1_manual_eq),
    "img1_cv_eq": compute_dct2(img1_cv_eq),
    "img2_original": compute_dct2(img2),
    "img2_manual_eq": compute_dct2(img2_manual_eq),
    "img2_cv_eq": compute_dct2(img2_cv_eq)
}

# Return everything for visualization
(img1, img2, img1_manual_eq, img2_manual_eq, img1_cv_eq, img2_cv_eq, histograms, dct2_images)

# Visualize histograms and DCT2 spectra
def plot_results(img_name, original, manual_eq, cv_eq, histograms, dct2_images):
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))

    axs[0, 0].imshow(original, cmap='gray')
    axs[0, 0].set_title(f'{img_name} Original')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(manual_eq, cmap='gray')
    axs[0, 1].set_title(f'{img_name} Manual Histogram Equalization')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(cv_eq, cmap='gray')
    axs[0, 2].set_title(f'{img_name} OpenCV Histogram Equalization')
    axs[0, 2].axis('off')

    axs[0, 3].axis('off')  # Unused

    axs[1, 0].plot(histograms[f'{img_name.lower()}_original'])
    axs[1, 0].set_title('Original Histogram')

    axs[1, 1].plot(histograms[f'{img_name.lower()}_manual_eq'])
    axs[1, 1].set_title('Manual EQ Histogram')

    axs[1, 2].plot(histograms[f'{img_name.lower()}_cv_eq'])
    axs[1, 2].set_title('OpenCV EQ Histogram')

    axs[1, 3].axis('off')  # Unused

    axs[2, 0].imshow(np.log(np.abs(dct2_images[f'{img_name.lower()}_original']) + 1), cmap='gray')
    axs[2, 0].set_title('Original DCT2 Spectrum')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(np.log(np.abs(dct2_images[f'{img_name.lower()}_manual_eq']) + 1), cmap='gray')
    axs[2, 1].set_title('Manual EQ DCT2 Spectrum')
    axs[2, 1].axis('off')

    axs[2, 2].imshow(np.log(np.abs(dct2_images[f'{img_name.lower()}_cv_eq']) + 1), cmap='gray')
    axs[2, 2].set_title('OpenCV EQ DCT2 Spectrum')
    axs[2, 2].axis('off')

    axs[2, 3].axis('off')  # Unused

    plt.tight_layout()
    plt.show()

# Plot for both images
plot_results("Img1", img1, img1_manual_eq, img1_cv_eq, histograms, dct2_images)
plot_results("Img2", img2, img2_manual_eq, img2_cv_eq, histograms, dct2_images)
