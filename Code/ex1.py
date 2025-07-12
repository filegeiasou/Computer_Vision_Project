import cv2
import numpy as np
import matplotlib.pyplot as plt

def mean_filter(img, ksize):
    """Εφαρμογή μάσκας μέσου όρου"""
    return cv2.blur(img, (ksize, ksize))

def add_awgn_noise(img, snr_db):
    """Προσθήκη AWGN με βάση το SNR που δίνεται""" 
    img = img.astype(np.float32) 
    # Υπολογισμός ισχύος σήματος 
    signal_power = np.mean(img ** 2) 
    # Από db σε γραμμική κλίμακα
    snr_linear = 10 ** (snr_db / 10.0)
    # Υπολογισμός ισχύος θορύβου με βάση το SNR
    noise_power = signal_power  / snr_linear

    """ Δημιουργία θορύβου AWGN"""
    # κανονική κατανομή θορύβου με μέσο 0 και τυπική απόκλιση sqrt(ισχύος θορύβου)
    noise = np.random.normal(0, np.sqrt(noise_power), img.shape)
    # Προσθήκη του θορύβου στην εικόνα
    noisy_img = img + noise
    # επειδη η αρχική εικονα ειναι σε grayscale, επιβεβαιωνόμαστε οτι ειναι σε εύρος [0, 255]
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img.astype(np.uint8)

def compute_mse(img1, img2):
    """Υπολογισμός Μέσου Τετραγωνικού Σφάλματος απο την αρχική και την φιλτραρισμένη εικόνα"""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def plot_image(img, title):
        # Plot με matplotlib
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()
    
def main():
    # === Φόρτωση της εικόνας ===
    og_img = cv2.imread('Code/Images/flowers.jpg') 
    # Ελέγχουμε αν η εικόνα φορτώθηκε σωστά
    if og_img is None:
        print("Δεν βρέθηκε η εικόνα")
        return
    # Μετατροπή σε grayscale αν χρειάζεται
    og_img = cv2.cvtColor(og_img, cv2.COLOR_BGR2GRAY) 

    ksizes = [5, 7, 9]  # τα διαφορετικά μήκοι μάσκας
    snrs = [10, 15, 18] # τα διαφορετικά SNRs σε dB
    res = {} # αποτελέσματα

    # === Βρόχος για κάθε SNR και μέγεθος μάσκας ===
    i = 0
    for snr in snrs:
        noisy_img = add_awgn_noise(og_img, snr)
        for k in ksizes:
            # Εφαρμογή μάσκας μέσου όρου και υπολογισμός MSE
            filter_img = mean_filter(noisy_img, k)
            mse = compute_mse(og_img, filter_img)

            # Εμφάνιση αποτελεσμάτων 
            res[i] = f"SNR={snr}db, Size={k}, MSE: {mse:.2f}"
            plot_image(filter_img, res[i])
            i += 1       
    # === Εμφάνιση αποτελεσμάτων σε κείμενο ===
    for k in res:
        print(res[k])

if __name__ == "__main__":
    main()