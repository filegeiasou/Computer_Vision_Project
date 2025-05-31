import cv2
import numpy as np
import matplotlib.pyplot as plt

def manual_hist_eq(img):
    # Υπολογισμός ιστογράμματος
    hist, bins = np.histogram(img.flatten(), 256, [0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # Εξίσωση ιστογράμματος
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img_eq = cdf[img]
    return img_eq

def plot_hist_and_dct(img, title):
    # Ιστόγραμμα
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.title(f"Ιστόγραμμα - {title}")
    plt.hist(img.flatten(),256,[0,256], color='gray')
    plt.xlim([0,256])
    # DCT2
    dct = cv2.dct(np.float32(img)/255.0)
    plt.subplot(1,2,2)
    plt.title(f"DCT2 - {title}")
    plt.imshow(np.log(np.abs(dct)+1), cmap='gray')
    plt.colorbar()
    plt.show()

for fname in ['im1.jpg', 'im2.jpg']:
    img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Δεν βρέθηκε η εικόνα {fname}")
        continue

    # Αρχικό ιστόγραμμα και DCT2
    plot_hist_and_dct(img, f"{fname} - αρχική")

    # Εξίσωση με δικό μας κώδικα
    img_eq_manual = manual_hist_eq(img)
    plot_hist_and_dct(img_eq_manual, f"{fname} - manual hist eq")

    # Εξίσωση με OpenCV
    img_eq_cv = cv2.equalizeHist(img)
    plot_hist_and_dct(img_eq_cv, f"{fname} - cv2.equalizeHist")

    # Εμφάνιση εικόνων για σύγκριση
    cv2.imshow(f"{fname} - αρχική", img)
    cv2.imshow(f"{fname} - manual hist eq", img_eq_manual)
    cv2.imshow(f"{fname} - cv2.equalizeHist", img_eq_cv)
    cv2.waitKey(0)
    cv2.destroyAllWindows()