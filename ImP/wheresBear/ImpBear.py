from ImP.imageProcessing.imageProcessingInterface import ImageProcessor
from ImP.wheresBear.wheresBearGenerator import generateDatabase
import sys
import glob
import os

"""
Can be called with:
python3 -m ImP.wheresBear.ImpBear numImages
"""

sizeOfDB = int(sys.argv[1])
wheres_bear_dir = os.path.dirname(__file__)
print(wheres_bear_dir)
markers = glob.glob(os.path.join(wheres_bear_dir, 'marker_*.png'))
backgrounds = [os.path.join(wheres_bear_dir, 'background.png')]

generateDatabase(size=sizeOfDB, quarry_images=markers, backgrounds=backgrounds)
images = glob.glob(os.path.join(wheres_bear_dir, 'database', 'Rot*_Pos*_Siz*_skew*'))
print("images is")
print(images)
f = open(os.path.join(wheres_bear_dir, 'results.txt'), 'w')
for image in images:
    try:
        imp = ImageProcessor(image)
        (corners, marker_ids) = imp._find_fiducial_markers()
        if marker_ids is None:
            f.write('found {0} at {1} in file {2}\n'.format(markers, corners, image))
        else:
            f.write('did not find {0} at {1} in file {2}\n'.format(markers, corners, image))
    except Exception as ex:
        print("Warning! Exception: {}".format(str(ex)))
        f.write('problem with {}/n'.format(image))
f.close()
