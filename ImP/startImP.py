from ImP.imageProcessing.imageProcessingInterface import ImageProcessor

# this file can be called from the python3 shell as such:
# exec(open("ImP/startImP.py").read())

img_path = "ImP/imageProcessing/test_files/2_ZENITH_0_BACK.jpg"
imp = ImageProcessor(img_path)
rvecs, tvecs = imp._find_pose()
