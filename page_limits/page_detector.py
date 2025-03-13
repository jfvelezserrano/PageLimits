import numpy as np
import cv2
from matplotlib import pyplot as plt

class Preprocesor:
    def process(self,image):
        pass

class OtsuPreprocesor(Preprocesor):
    def __binarize_image(image,verbose = False):        
        _,binary_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binary_image


class HistPreprocesor(Preprocesor):
    def __binarize_image(image,verbose = False):        
        hist_item = cv2.calcHist([image],[0],None,[256],[0,255])
        if verbose:
            plt.plot(hist_item.reshape(256,))
            plt.show()

        hstograma_suavizado = gaussian_filter1d(hist_item.reshape(256,),sigma=10)
        if verbose:
            plt.plot(hstograma_suavizado)
            plt.show()

        maximos = find_peaks(hstograma_suavizado,distance=10,height=image.shape[0]*image.shape[1]*0.001)

        if len(maximos[0]) > 1:
            threshold = (maximos[0][-1] + maximos[0][-2]) / 2
        else:
            threshold = 150

        if verbose:
            print(threshold)
        binary_image = ((image > threshold)*255).astype(np.uint8)
        
        if verbose:
            plt.imshow(image,cmap="gray")
            plt.show()
        return binary_image
        
        
class AdaptativePreprocesor(Preprocesor):
    def __binarize_image(image,verbose = False):        
        #_,binary_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        #return binary_image

        binary_image = cv2.adaptiveThreshold(image,255,cv2.BORDER_REPLICATE,cv2.THRESH_BINARY,31,5)
        return binary_image
        
        hist_item = cv2.calcHist([image],[0],None,[256],[0,255])
        if verbose:
            plt.plot(hist_item.reshape(256,))
            plt.show()

        hstograma_suavizado = gaussian_filter1d(hist_item.reshape(256,),sigma=10)
        if verbose:
            plt.plot(hstograma_suavizado)
            plt.show()

        maximos = find_peaks(hstograma_suavizado,distance=10,height=image.shape[0]*image.shape[1]*0.001)

        if len(maximos[0]) > 1:
            threshold = (maximos[0][-1] + maximos[0][-2]) / 2
        else:
            threshold = 150

        if verbose:
            print(threshold)
        binary_image = ((image > threshold)*255).astype(np.uint8)
        
        if verbose:
            plt.imshow(image,cmap="gray")
            plt.show()
        return binary_image

class PointDetector:
    pass

class PointFilter:
    pass


class PageDetector:
    #def __init__(self,preprocesor:Preprocesor, pointDetector:PointDetector, pointFilter:PointFilter):
    #    pass
    
    def __init__(self):
        self.preprocesor = Preprocesor()
        self.pointDetector = PointDetector()
        self.pointFilter = PointFilter()

    def detect(self, image:np.ndarray):
        preprocessed_image = self.preprocesor.process(image)
        points = self.pointDetector.getPoints(preprocessed_image)
        filtered_points = self.pointFilter.filter(points)        
        return filtered_points