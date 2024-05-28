#criar criar softawre em python combinando em sequência, no mínimo, 
#3 tipos diferentes de ruído, 
#3 tipos diferentes de técnica de suavização 
# 3 tipos diferentes de detectores de borda. 

#Explore também os parâmetros de cada técnica e dos ruídos

import cv2
import numpy as np

# Load an image
img = cv2.imread('flor.jpeg', cv2.IMREAD_GRAYSCALE)

# Generate Gaussian noise
gauss = np.random.normal(0,0.5,img.size)
gauss = gauss.reshape(img.shape[0],img.shape[1]).astype('uint8')
img_gauss = cv2.add(img,gauss)

# Generate Salt-and-pepper noise
s_p = np.copy(img)
num_salt = np.ceil(0.05 * img.size)
coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
s_p[coords] = 255

# Generate Poisson noise
vals = len(np.unique(img))
vals = 2 ** np.ceil(np.log2(vals))
poisson = np.random.poisson(img * vals) / float(vals)
poisson = poisson.astype('uint8')

# Apply Gaussian blur
blur_gauss = cv2.GaussianBlur(img_gauss,(5,5),0)

# Apply Median blur
blur_median = cv2.medianBlur(s_p, 5)

# Apply Bilateral filter
blur_bilateral = cv2.bilateralFilter(poisson,9,75,75)

# Apply Sobel edge detection
edges_sobel = cv2.Sobel(blur_gauss,cv2.CV_64F,1,1,ksize=5)

# Apply Laplacian edge detection
edges_lap = cv2.Laplacian(blur_median,cv2.CV_64F)

# Apply Canny edge detection
edges_canny = cv2.Canny(blur_bilateral,100,200)

# Display the images
cv2.imshow('Original', img)
cv2.imshow('Gaussian Noise', img_gauss)
cv2.imshow('Salt and Pepper Noise', s_p)
cv2.imshow('Poisson Noise', poisson)
cv2.imshow('Gaussian Blur', blur_gauss)
cv2.imshow('Median Blur', blur_median)
cv2.imshow('Bilateral Filter', blur_bilateral)
cv2.imshow('Sobel Edge Detection', edges_sobel)
cv2.imshow('Laplacian Edge Detection', edges_lap)
cv2.imshow('Canny Edge Detection', edges_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
