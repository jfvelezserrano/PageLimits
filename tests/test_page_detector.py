import unittest

import cv2
import numpy as np
from page_limits.page_detector import PageDetector

class TestPageDetector(unittest.TestCase):
    """
    Test
    """
    
    
    def compare_corners(self, corner_list_1, corner_list_2):
        detected_corners = set()
        for corner_1 in corner_list_1:
            best_corner = None
            for corner_2 in corner_list_2:
                if np.linalg.norm(corner_1 - corner_2) < 100:
                    best_corner = corner_2
            detected_corners.add(tuple(best_corner))
        self.assertEqual(len(detected_corners),4)
    
    
    def test_page_detection(self):
        p = PageDetector(False)
        image = cv2.imread("samples/learning/Hoja_02.jpg",0)
        result = p.detect(image)

        expected_result = [[211,351],[2565,452],[195,3751],[2574,3670]]
        expected_result = np.array(expected_result)

        self.compare_corners(result,expected_result)
        
if __name__ == '__main__':
    unittest.main()
