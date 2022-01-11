import cv2
import numpy as np
from skimage.filters import median

class Preprocessing:

    @staticmethod
    def preprocess_photo(image):
        grey_img = Preprocessing.grey_scale(image)
        nonnoise_img = Preprocessing.remove_noise(grey_img)
        equilized_img = Preprocessing.equilize_hist(nonnoise_img)
        return equilized_img,grey_img
        
    @staticmethod
    def equilize_hist(image):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        equalized_img = clahe.apply(image.astype('uint8'))
        return equalized_img

    @staticmethod
    def remove_noise(image):
        return cv2.bilateralFilter(image.astype('uint8'), 11, 17, 17)
    @staticmethod
    def grey_scale(image):

        grey_image = np.zeros((image.shape[0],image.shape[1]))
        grey_image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]

        return grey_image