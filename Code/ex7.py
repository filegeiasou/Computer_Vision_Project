import cv2
import numpy as np
import matplotlib.pyplot as plt

# Butterworth high-pass φίλτρο
def butterworth_high_pass_filter(shape, cutoff, order):
    P, Q = shape 
    # Δημιουργία πλέγματος συχνοτήτων
    u = np.arange(P)
    v = np.arange(Q)
    # 2D συστημα συντεταγμένων
    U, V = np.meshgrid(v, u)
    # Υπολογισμός Ευκλειδιας αποστάσης από το κέντρο 
    D_uv = np.sqrt((U - Q/2)**2 + (V - P/2)**2)
    # Υπολογισμός συνάρτησης μεταφοράς Butterworth φίλτρου
    H = 1 / (1 + (cutoff / (D_uv + 1e-5))**(2 * order))
    return H

def homomorphic_filter(img, gamma_l=0.5, gamma_h=2.0, cutoff=30, order=1):
    img = img.astype(np.float32) # για να αποφύγουμε overflow
    img += 1  # για αποφυγή log(0)
    log_img = np.log(img) # μετατροπή σε log

    # Υπολογισμός DFT
    dft = np.fft.fft2(log_img)
    # Μετατόπιση της μηδενικής συχνότητας στο κέντρο
    dft_shift = np.fft.fftshift(dft)

    # Εφαρμογή Butterworth high-pass φίλτρου πρώτης τάξης 
    H = butterworth_high_pass_filter(img.shape, cutoff, order)
    # Διορθωση gamma_l (illuminance -> φωτεινότητα) και gamma_h (high frequency -> λεπτομέρειες)
    H = gamma_l + (gamma_h - gamma_l) * H

    # Εφαρμογή φίλτρου στο DFT
    filtered_dft = dft_shift * H
    # Αντιστροφή μετατόπιση
    dft_inv = np.fft.ifftshift(filtered_dft)
    # Αντίστροφο FFT για επιστροφή στο χωρικό πεδίο
    img_back = np.fft.ifft2(dft_inv)  
    # Αντιστροφο log + 1 offset
    img_back = np.exp(np.real(img_back)) - 1  

    return img_back

def main():
    # === Φόρτωση και προεπεξεργασία εικόνας ===
    img_path = 'Code/Images/car.jpg'
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    # Ελέγχουμε αν η εικόνα φορτώθηκε σωστά
    if img is None:
        print("Δεν βρέθηκε η εικόνα")
        return
    filter_img = homomorphic_filter(img)

    # === Εμφάνιση αποτελεσμάτων ===
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title('Original')
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Homomorphic Filtered')
    plt.imshow(filter_img, cmap='gray')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()