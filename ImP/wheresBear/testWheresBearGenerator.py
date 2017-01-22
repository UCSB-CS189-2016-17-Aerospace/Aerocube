import unittest
import wheresBearGenerator

class TestSubFunctions(unittest.TestCase):
    marker0='marker_4X4_sp6_id7.png'
    TESTIMAGENAME='marker_4X4_sp6_id7'

    def testCreateImage(self):
        wheresBearGenerator.createImage(quarry_image_name=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(500,500)),
                                        size=20)

    def testCreateImageResizeandRotate(self):
        wheresBearGenerator.createImage(quarry_image_name=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(500,500)),
                                        rotation_degree=45,
                                        size=20)
    def testCreateImageSkew(self):
        wheresBearGenerator.createImage(quarry_image_name=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(500,500)),
                                        size=20,
                                        skew_values=[20,0,-20,0,0,0,0,0])

    def testCreateImagePos(self):
        wheresBearGenerator.createImage(quarry_image_name=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(500,500)),
                                        size=20,pos=(200,200))