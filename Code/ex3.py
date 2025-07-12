import cv2
import numpy as np
import matplotlib.pyplot as plt

def sharpen_image(img):
    """Κλασικός πυρήνας όξυνσης (unsharp mask) """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(img, -1, kernel)

def compute_mse(img1, img2):
    """Υπολογισμός Μέσου Τετραγωνικού Σφάλματος απο την αρχική και την φιλτραρισμένη εικόνα"""
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def plot_images(og, sharp):
    """Εμφάνιση των εικόνων"""
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(og, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title("Sharpened")
    plt.imshow(sharp, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    img_paths = [
        'Images/bridge.tif',
        'Images/im1.jpg',
        'Images/im2.jpg'
    ]
    res = {}
    i = 0
    # === Βρόχος για κάθε εικόνα ===
    for fname in img_paths:
        # === Φόρτωση εικόνας ===
        img = cv2.imread(fname)
        # αν η εικόνα δεν βρέθηκε, συνεχίζουμε στην επόμενη
        if img is None:
            print(f"Δεν βρέθηκε η εικόνα {fname}")
            continue
        # Μετατροπή σε grayscale αν χρειάζεται
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               

        # Εφαρμογή όξυνσης με unsharp mask
        sharp = sharpen_image(img_gray)
        # Υπολογισμός MSE
        mse = compute_mse(img_gray, sharp)

        # Εμφάνιση και αποθήκευση
        res[i] = f"{fname}: MSE = {mse:.2f}"
        # Εμφάνιση εικόνων
        plot_images(img_gray, sharp)
        i += 1
    # === Εμφάνιση αποτελεσμάτων ===
    for k in res:
        print(res[k])

if __name__ == "__main__":
    main()