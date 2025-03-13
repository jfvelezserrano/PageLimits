import unittest

import cv2
from page_limits.page_detector import PageDetector

class TestPageDetector(unittest.TestCase):
    
    def test_page_detection(self):
        p = PageDetector()
        image = cv2.imread("images/page_01.jpeg",0)
        result = p.detect(image)
        expected_result = [(195, 164), (484, 172), (210, 888), (506, 875)]
        self.assertEqual(result,expected_result)
        
if __name__ == '__main__':
    unittest.main()