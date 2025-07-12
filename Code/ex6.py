import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from pathlib import Path

# Helper function για τον υπολογισμό histogram equalization με OpenCV
def compute_histogram(image):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    return hist.flatten()

# Helper function για την δικίας μας συνάρτηση histogram equalization
def manual_hist_equalization(image):
    hist = compute_histogram(image)
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    equalized_img = np.interp(image.flatten(), range(256), cdf_normalized).reshape(image.shape).astype(np.uint8)
    return equalized_img

# Helper function για τον υπολογισμό DCT2
def compute_dct2(image):
    return dct(dct(image.T, norm='ortho').T, norm='ortho')

# Helper functions για την εμφάνιση αρχικών εικόνων, ιστογραμμάτων και DCT2
def plot_image(img, title, axs, col, row=0):
    axs[row, col].imshow(img, cmap='gray')
    axs[row, col].set_title(title)
    axs[row, col].axis('off')

def plot_histogram(hist, name, title, axs, col, row=1):
    axs[row, col].plot(hist[f'{name}'])
    axs[row, col].set_title(title)

def plot_dct2_spectrum(dct2_image, title, axs, col, row=2):
    axs[row, col].imshow(np.log(np.abs(dct2_image) + 1), cmap='gray')
    axs[row, col].set_title(title)
    axs[row, col].axis('off')

# Αρχικές εικόνες, Ιστογράμματα και DCT2 σε 3x3 grid 
def plot_results(img_name, original, manual_eq, cv_eq, histograms, dct2_images):
    # Δημιουργία υποπινάκων για την εμφάνιση των plots
    fig, axs = plt.subplots(3, 3, figsize=(20, 12))

    # Εμφάνιση αρχικής εικόνας, με manual EQ και με OpenCV EQ
    plot_image(original, f"{img_name} Original", axs, 0)
    plot_image(manual_eq, f'{img_name} Manual Histogram Equalization', axs, 1)
    plot_image(cv_eq, f'{img_name} OpenCV Histogram Equalization', axs, 2)
    
    ## Εμφάνιση αρχικού ιστογράμματος, manual EQ και OpenCV EQ
    plot_histogram(histograms, f'{img_name}_original', 'Original Histogram', axs, 0)
    plot_histogram(histograms, f'{img_name}_manual_eq','Manual EQ Histogram', axs, 1)
    plot_histogram(histograms, f'{img_name}_cv_eq','OpenCV EQ Histogram', axs, 2)

    # Εμφάνιση DCT2 για την αρχική εικόνα, manual EQ και OpenCV EQ
    plot_dct2_spectrum(dct2_images[f'{img_name}_original'], 'Original DCT2 Spectrum', axs, 0)
    plot_dct2_spectrum(dct2_images[f'{img_name}_manual_eq'], 'Manual EQ DCT2 Spectrum', axs, 1)
    plot_dct2_spectrum(dct2_images[f'{img_name}_cv_eq'], 'OpenCV EQ DCT2 Spectrum', axs, 2)

    plt.tight_layout() # για να μην επικαλύπτονται τα plots
    plt.show()

def main():
    # === Φόρτωση εικόνων ===
    img1 = cv2.imread("Images/im1.jpg")
    img2 = cv2.imread("Images/im2.jpg")

    # Ελέγχουμε αν οι εικόνες φορτώθηκαν σωστά
    if img1 is None or img2 is None:
        print("Δεν βρέθηκαν οι εικόνες")
        return

    # Μετατροπή σε grayscale αν χρειάζεται
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  

    # Εφαρμογή histogram equalization με δικιά μας συνάρτηση και με OpenCV
    img1_manual_eq = manual_hist_equalization(img1)
    img2_manual_eq = manual_hist_equalization(img2)
    img1_cv_eq = cv2.equalizeHist(img1)
    img2_cv_eq = cv2.equalizeHist(img2)

    # Υπολογισμός ιστογραμμάτων για κάθε εικόνα
    histograms = {
        "img1_original": compute_histogram(img1),
        "img1_manual_eq": compute_histogram(img1_manual_eq),
        "img1_cv_eq": compute_histogram(img1_cv_eq),
        "img2_original": compute_histogram(img2),
        "img2_manual_eq": compute_histogram(img2_manual_eq),
        "img2_cv_eq": compute_histogram(img2_cv_eq)
    }

    # Υπολογισμός DCT2 για κάθε εικόνα
    dct2_images = {
        "img1_original": compute_dct2(img1),
        "img1_manual_eq": compute_dct2(img1_manual_eq),
        "img1_cv_eq": compute_dct2(img1_cv_eq),
        "img2_original": compute_dct2(img2),
        "img2_manual_eq": compute_dct2(img2_manual_eq),
        "img2_cv_eq": compute_dct2(img2_cv_eq)
    }

    # Εμφάνιση αποτελεσμάτων και για τις δύο εικόνες
    plot_results("img1", img1, img1_manual_eq, img1_cv_eq, histograms, dct2_images)
    plot_results("img2", img2, img2_manual_eq, img2_cv_eq, histograms, dct2_images) 

if __name__ == "__main__":
    main()