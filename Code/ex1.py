import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_awgn_noise(image, snr_db):
    """Προσθήκη θορύβου AWGN με βάση το επιθυμητό SNR σε dB"""
    image = image.astype(np.float32)
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def mean_filter(image, ksize):
    """Εφαρμογή μάσκας μέσου όρου"""
    return cv2.blur(image, (ksize, ksize))

def compute_mse(image1, image2):
    """Υπολογισμός Μέσου Τετραγωνικού Σφάλματος"""
    return np.mean((image1.astype(np.float32) - image2.astype(np.float32)) ** 2)

# === Φόρτωση της εικόνας ===
original = cv2.imread('flowers.jpg')
original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)  # για απλοποίηση (γκρι)

ksizes = [5, 7, 9]
snrs = [10, 15, 18]

results = {}

# === Βρόχος για κάθε SNR και μέγεθος μάσκας ===
for snr in snrs:
    noisy = add_awgn_noise(original, snr)
    for k in ksizes:
        filtered = mean_filter(noisy, k)
        mse = compute_mse(original, filtered)
        key = f"SNR={snr}dB, Kernel={k}x{k}"
        results[key] = mse

        # Αποθήκευση ή εμφάνιση (προαιρετικά)
        print(f"{key} -> MSE: {mse:.2f}")
        cv2.imshow(key, filtered)
        cv2.waitKey(0)

cv2.destroyAllWindows()
