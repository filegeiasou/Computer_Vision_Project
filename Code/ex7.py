import cv2
import numpy as np
import matplotlib.pyplot as plt

def butterworth_high_pass_filter(shape, cutoff, order):
    P, Q = shape
    u = np.arange(P)
    v = np.arange(Q)
    U, V = np.meshgrid(v, u)
    D_uv = np.sqrt((U - Q/2)**2 + (V - P/2)**2)
    H = 1 / (1 + (cutoff / (D_uv + 1e-5))**(2 * order))
    return H

def homomorphic_filter(image, gamma_l=0.5, gamma_h=2.0, cutoff=30, order=1):
    image = image.astype(np.float32)
    image += 1  # για αποφυγή log(0)
    log_image = np.log(image)

    dft = np.fft.fft2(log_image)
    dft_shift = np.fft.fftshift(dft)

    H = butterworth_high_pass_filter(image.shape, cutoff, order)
    H = gamma_l + (gamma_h - gamma_l) * H

    filtered_dft = dft_shift * H
    dft_inv = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(dft_inv)
    img_back = np.exp(np.real(img_back)) - 1  # inverse log + 1 offset

    img_back = np.clip(img_back, 0, 255)
    return img_back.astype(np.uint8)

# === Φόρτωση και προεπεξεργασία εικόνας ===
image = cv2.imread('Code/Images/input/car.jpg', cv2.IMREAD_GRAYSCALE)
filtered_image = homomorphic_filter(image)
# Αποθήκευση εικόνας 
cv2.imwrite('Code/Images/output/car.jpg', filtered_image)

# === Εμφάνιση αποτελεσμάτων ===
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Αρχική Εικόνα')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Μετά από Ομοιομορφικό Φιλτράρισμα')
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')
plt.show()
