import unittest
import wheresBearGenerator

class TestSubFunctions(unittest.TestCase):
    marker0=wheresBearGenerator.Image.open('marker_4X4_sp6_id7.png')

    def testCreateImage(self):
        wheresBearGenerator.createImage(quarry_image=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(100,100)),
                                        name='test0.png',
                                        size=10)

    def testCreateImageResizeandRotate(self):
        wheresBearGenerator.createImage(quarry_image=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(100,100)),
                                        name='test1.png',
                                        rotation_degree=45,
                                        size=10)
    def testCreateImageSkew(self):
        wheresBearGenerator.createImage(quarry_image=self.marker0,
                                        background=wheresBearGenerator.Image.new("RGB",(100,100)),
                                        name='test2.png',
                                        size=10,
                                        skew_values=[20,0,-20,0,0,0,0,0])