import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import maximum_filter
from scipy.signal import find_peaks

class Preprocesor:
    
    def __init__(self, verbose = False):
        self.verbose = verbose
    
    def process(self,image)->np.ndarray:
        pass
    
class OtsuPreprocesor(Preprocesor):
    
    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def __binarize_image(self, image):
        _,binary_image = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return binary_image
    
    def __morphology(self, image):
        image = cv2.erode(image,kernel=np.ones((11,11)))
        image = cv2.dilate(image,kernel=np.ones((11,11)))
        image = cv2.dilate(image,kernel=np.ones((33,33)))
        image = cv2.erode(image,kernel=np.ones((33,33)))
        if self.verbose:
            plt.imshow(image,cmap="gray")
            plt.show()
        return image        
    
    def process(self,image):
        image = self.__binarize_image(image)
        image = self.__morphology(image)
        return image

class HistPreprocesor(Preprocesor):
    
    def __init__(self, verbose = False):
        self.verbose = verbose
    
    def __binarize_image(self, image):
        hist_item = cv2.calcHist([image],[0],None,[256],[0,255])
        if self.verbose:
            plt.plot(hist_item.reshape(256,))
            plt.show()

        hstograma_suavizado = gaussian_filter1d(hist_item.reshape(256,),sigma=10)
        if self.verbose:
            plt.plot(hstograma_suavizado)
            plt.show()

        maximos = find_peaks(hstograma_suavizado,distance=10,height=image.shape[0]*image.shape[1]*0.001)

        if len(maximos[0]) > 1:
            threshold = (maximos[0][-1] + maximos[0][-2]) / 2
        else:
            threshold = 150

        if self.verbose:
            print(threshold)
        binary_image = ((image > threshold)*255).astype(np.uint8)
        
        if self.verbose:
            plt.imshow(image,cmap="gray")
            plt.show()
            
        return binary_image

    def __morphology(self, image):
        image = cv2.erode(image,kernel=np.ones((11,11)))
        image = cv2.dilate(image,kernel=np.ones((11,11)))
        image = cv2.dilate(image,kernel=np.ones((33,33)))
        image = cv2.erode(image,kernel=np.ones((33,33)))
        if self.verbose:
            plt.imshow(image,cmap="gray")
            plt.show()
        return image        
        
    def process(self,image):
        image = self.__binarize_image(image)
        image = self.__morphology(image)
        return image
        
class AdaptativePreprocesor(Preprocesor):
    def __binarize_image(self, image):
        binary_image = cv2.adaptiveThreshold(image,255,cv2.BORDER_REPLICATE,cv2.THRESH_BINARY,31,5)
        return binary_image

    def process(self,image):
        image = self.__binarize_image(image)
        return image
    
    def __init__(self, verbose=False):
        self.verbose = verbose

class PointDetector:
    pass

class HarrisPointDetector(PointDetector):
    
    def __get_harris_points(self, image):
        blockSize = 31 # Tamaño de la ventana W
        ksize = 31 #Tamaño del kernel de derivación
        k = 0.05 #Factor de harris
        esquinas = cv2.cornerHarris(image,blockSize,ksize,k)

        umbral = 0.1 * esquinas.max()
        maximos_locales = (esquinas == maximum_filter(esquinas, size=10)) & (esquinas > umbral)
        
        puntos_harris = np.argwhere(maximos_locales)
        puntos_harris = np.array(puntos_harris)
        puntos_harris_copia = puntos_harris.copy()
        puntos_harris[:,0] = puntos_harris_copia[:,1]
        puntos_harris[:,1] = puntos_harris_copia[:,0]
        if self.verbose:
            print(puntos_harris.shape)
        
        return puntos_harris

    def __init__(self, verbose = False):
        self.verbose = verbose
    
    def detect(self,image):
        return self.__get_harris_points(image)

class PointFilter:
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.KEYPOINT_SIZE = 31
        self.RESIZE = 4
    
    def filter(self,points):
        pass
    
    def get_resize_factor(self):
        return self.RESIZE


class SiftPointFilter(PointFilter):
    
    def __str_2_tuple(self, cadena):
        # Eliminar espacios en blanco y saltos de línea
        cadena = cadena.strip()
        # Dividir la cadena por la coma y convertir a enteros
        try:
            tupla = tuple(map(int, cadena.split(",")))
            if len(tupla) == 2:
                return (tupla[0]//self.RESIZE,tupla[1]//self.RESIZE)
            else:
                raise ValueError("La cadena no contiene exactamente dos valores separados por coma.")
        except ValueError as e:
            print(f"Error al convertir la cadena a tupla: {e}")
            return None    
    
    def __read_samples(self):
        samples = []

        with open("samples/learning/esquinas.txt") as file:
            file_name = file.readline()
            while file_name != "":
                p1 = self.__str_2_tuple(file.readline())
                p2 = self.__str_2_tuple(file.readline())
                p3 = self.__str_2_tuple(file.readline())
                p4 = self.__str_2_tuple(file.readline())
                samples.append([file_name[:-1],p1,p2,p3,p4])
                file_name = file.readline()
        return samples
        
        
    def __compute_refence_descriptors(self, samples):
        """
            Por cada linea en samples
            Carga la imagen
            La preprocesa
            Calcula los descriptores
            Almacena los descriptores
        """

        reference_descriptors = []

        for image_gt in samples:
            image = cv2.imread("samples/learning/"+image_gt[0],0)
            image = cv2.resize(image,(image.shape[1]//self.RESIZE,image.shape[0]//self.RESIZE))
            
            preprocess_image = self.preprocesor.process(image)
            
            corner_matrix = np.array(image_gt[1:])
            
            if self.verbose:
                plt.subplot(1,4,1)
                plt.imshow(image,cmap="gray")
                plt.subplot(1,4,3)
                plt.imshow(preprocess_image,cmap="gray")
                plt.subplot(1,4,4)
                plt.imshow(image,cmap="gray")
                plt.scatter(corner_matrix[:, 0], corner_matrix[:, 1], c='red', marker="+")

            corner_keypoints = [cv2.KeyPoint(float(x[0]), float(x[1]), self.KEYPOINT_SIZE) for x in corner_matrix]
            corner_descriptors = self.sift.compute(preprocess_image,corner_keypoints)
                        
            if self.verbose:
                print(corner_descriptors[1].shape)
                plt.show()
                
            reference_descriptors.append(corner_descriptors[1])
                            
        reference_descriptors = np.array(reference_descriptors)
        num_images, corners, features = reference_descriptors.shape
        reference_descriptors = reference_descriptors.reshape(num_images * corners, features)
        return reference_descriptors    
    
    def __init__(self, preprocesor:Preprocesor, verbose = False):
        super().__init__(verbose)
        self.preprocesor = preprocesor
        self.sift = cv2.xfeatures2d.SIFT_create()

        samples = self.__read_samples()
        self.reference_descriptors = self.__compute_refence_descriptors(samples)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        
    def filter(self,points, image, verbose=False):
        harris_keypoints = [cv2.KeyPoint(float(x[0]), float(x[1]), self.KEYPOINT_SIZE) for x in points]
        query_descriptors = self.sift.compute(image,harris_keypoints)

        for cont in range(0,query_descriptors[1].shape[0]):
            matches = self.bf.knnMatch(query_descriptors[1], self.reference_descriptors, k=1) 
            
            height = image.shape[0]
            width = image.shape[1]
            
            best_matches_Q1 = (float("inf"),width//4,height//4)
            best_matches_Q2 = (float("inf"),width//4,3*height//4)
            best_matches_Q3 = (float("inf"),3*width//4,height//4)
            best_matches_Q4 = (float("inf"),3*width//4,3*height//4)
            
            for match in matches:
                distance = match[0].distance
                index = match[0].queryIdx
                coords = query_descriptors[0][index].pt
                if (coords[0] < width/2) and (coords[1] < height/2):
                    if best_matches_Q1[0] > distance:
                        best_matches_Q1 = (distance,coords[0],coords[1])
                elif (coords[0] < width/2) and (coords[1] >= height/2):
                    if best_matches_Q2[0] > distance:
                        best_matches_Q2 = (distance,coords[0],coords[1])
                elif (coords[0] >= width/2) and (coords[1] >= height/2):
                    if best_matches_Q4[0] > distance:
                        best_matches_Q4 = (distance,coords[0],coords[1])
                else:
                    if best_matches_Q3[0] > distance:
                        best_matches_Q3 = (distance,coords[0],coords[1])
            
            output = (best_matches_Q1,best_matches_Q2,best_matches_Q3,best_matches_Q4)
            output = np.array(output)
            output = output[:,1:3].astype(np.uint32)
            
            if True:#verbose:
                plt.imshow(image,cmap="gray")
                plt.scatter(output[:, 0], output[:, 1], c='red', marker="+")
                plt.show()
            
            return output    

class PageDetector:
    def __init__(self,preprocesor:Preprocesor, pointDetector:PointDetector, pointFilter:PointFilter, verbose=False):
        self.preprocesor = preprocesor
        self.pointDetector = pointDetector
        self.pointFilter = pointFilter
        self.verbose = verbose
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.preprocesor = AdaptativePreprocesor(self.verbose)
        self.pointDetector = HarrisPointDetector(self.verbose)
        self.pointFilter = SiftPointFilter(self.preprocesor, self.verbose)

    def detect(self, image:np.ndarray):
        RESIZE_FACTOR = self.pointFilter.get_resize_factor()
        image = cv2.resize(image,(image.shape[1]//RESIZE_FACTOR,image.shape[0]//RESIZE_FACTOR))
        preprocessed_image = self.preprocesor.process(image)
        points = self.pointDetector.detect(preprocessed_image)
        filtered_points = self.pointFilter.filter(points, image)        
        return filtered_points * self.pointFilter.get_resize_factor()