import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import skew, kurtosis
import os


GAUSSIAN_K = 5
MEDIAN_K = 5
AVERAGE_K = 5
BILATERAL_D = 9
BILATERAL_SC = 75
BILATERAL_SS = 75

LOWPASS_RATIO = 0.25
NOTCH_RADIUS = 12
PEAK_THRESHOLD = 0.85



def detect_noise(img):
    img = img.astype(np.float32)

    mean = np.mean(img)
    var = np.var(img)
    sk = skew(img.flatten())
    ku = kurtosis(img.flatten())

    # Salt & pepper detection
    salt = np.sum(img > 250)
    pepper = np.sum(img < 5)
    ratio = (salt + pepper) / img.size

    if ratio > 0.02:
        return "salt_pepper"


    f = np.fft.fftshift(np.fft.fft2(img))
    mag = np.log(1 + np.abs(f))
    mag_norm = mag / mag.max()
    peaks = np.sum(mag_norm > PEAK_THRESHOLD)

    if peaks > 30:
        return "periodic"

    # Speckle
    if abs(sk) > 1 and var / (mean + 1e-8) > 0.5:
        return "speckle"

    # Uniform noise
    if ku < 2:
        return "uniform"

    return "gaussian"



def spatial_filter(img, noise):
    if noise == "gaussian":
        return cv2.GaussianBlur(img,(GAUSSIAN_K,GAUSSIAN_K),0)

    if noise == "salt_pepper":
        return cv2.medianBlur(img,MEDIAN_K)

    if noise == "speckle":
        return cv2.bilateralFilter(img,BILATERAL_D,BILATERAL_SC,BILATERAL_SS)

    if noise == "uniform":
        return cv2.blur(img,(AVERAGE_K,AVERAGE_K))

    return img.copy()



def frequency_filter(img, noise):
    rows, cols = img.shape
    f = np.fft.fftshift(np.fft.fft2(img))
    r, c = rows//2, cols//2

    mask = np.ones((rows,cols),np.float32)

    if noise == "periodic":
        mag = np.log(1+np.abs(f))
        mag_norm = mag/mag.max()
        ys,xs = np.where(mag_norm>PEAK_THRESHOLD)

        for y,x in zip(ys,xs):
            if abs(y-r)>10 or abs(x-c)>10:
                cv2.circle(mask,(x,y),NOTCH_RADIUS,0,-1)

    else:
        cutoff = int(LOWPASS_RATIO*min(rows,cols))
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i-r)**2 + (j-c)**2)
                mask[i,j] = np.exp(-(d**2)/(2*(cutoff**2)))

    filtered = f*mask
    img_back = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return np.clip(img_back,0,255).astype(np.uint8)



def metrics(ref, test):
    ref = ref.astype(np.float32)
    test = test.astype(np.float32)

    mse = np.mean((ref-test)**2)
    psnr = np.inf if mse==0 else 10*np.log10(255**2/mse)
    s = ssim(ref,test,data_range=255)

    return mse,psnr,s



def show(img,sp,fr,title,noise):
    plt.figure(figsize=(14,4))
    plt.suptitle(f"{title} | Noise: {noise}")

    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray')
    plt.title("Noisy")

    plt.subplot(1,3,2)
    plt.imshow(sp,cmap='gray')
    plt.title("Spatial")

    plt.subplot(1,3,3)
    plt.imshow(fr,cmap='gray')
    plt.title("Frequency")

    for i in range(1,4):
        plt.subplot(1,3,i).axis('off')

    plt.show()



def main():

    os.makedirs("output/spatial",exist_ok=True)
    os.makedirs("output/frequency",exist_ok=True)

    images = [f"img_{i}.png" for i in range(1,8)]
    table = []

    for name in images:

        path = name
        if not os.path.exists(path):
            print("Missing:",name)
            continue

        img = cv2.imread(path,0)
        noise = detect_noise(img)

        sp = spatial_filter(img,noise)
        fr = frequency_filter(img,noise)

        cv2.imwrite(f"output/spatial/{name}",sp)
        cv2.imwrite(f"output/frequency/{name}",fr)

        _,ps_sp,ss_sp = metrics(img,sp)
        _,ps_fr,ss_fr = metrics(img,fr)

        table.append((name,noise,ps_sp,ps_fr,ss_sp,ss_fr))

        show(img,sp,fr,name,noise)



    print("\n---------------------------------------------------------------")
    print("Image   Noise        PSNR(S)  PSNR(F)  SSIM(S)  SSIM(F)")
    print("---------------------------------------------------------------")

    for r in table:
        print(f"{r[0]:<7} {r[1]:<11} {r[2]:>7.2f} {r[3]:>8.2f} {r[4]:>8.4f} {r[5]:>8.4f}")

    print("---------------------------------------------------------------")

if __name__=="__main__":
    main()
