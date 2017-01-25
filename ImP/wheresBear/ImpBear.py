from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.wheresBear.wheresBearGenerator import generateDatabase
import sys
import glob

sizeOfDB = sys.argv[1]
markers = glob.glob('marker_*.png')

generateDatabase(size=sizeOfDB,quarry_images=markers)
images = glob.glob('Rot*_Pos*_Siz*_skew*')
f = open('results.txt', 'w')
for image in images:
    try:
        imp = ImageProcessor(image)
        (corners, marker_ids) = imp._find_fiducial_markers()
        f.write('found {0} at {1} in file {2}/n'.format(markers, corners, image))
    except:
        f.write('problem with {}/n'.format(image))
f.close()
