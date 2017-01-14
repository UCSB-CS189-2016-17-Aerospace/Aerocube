import numpy
from PIL import Image
import random

def generateDatabase(size, quarry_images, backgrounds=[Image.new("RGB", (1000, 1000))]):
    #todo finish function
    for i in range(size):
        createImage(quarry_images[0], backgrounds[0], '%i' % (i))

def createImage(quarry_image, background, name, rotation_degree=0, pos=(0,0),size=1,skew_values=None):
    '''
    :param quarry_image: image of quarry
    :param background: background image
    :param name: name of image produced
    :param rotation_degree:
    :param pos: where to place the quarry images' top left corner when pasted defaults top left corner
    :param size: size scalar
    :param skew_values: 8 long array of ofsets for the corners of the quarry image
    :return:
    '''
    #quarry_image.rotate(rotation_degree)
    quarry_image=quarry_image.resize((quarry_image.width*size,quarry_image.height*size))
    if skew_values is not None:
        width,height=quarry_image.size
        corners=[(skew_values[0],skew_values[1]),
                 (width+skew_values[2],skew_values[3]),
                 (width+skew_values[4],height+skew_values[5]),
                 (skew_values[6],height-skew_values[7])]
        quarry_image=skew(quarry_image,corners)
    quarry_image=quarry_image.rotate(rotation_degree)

    background.paste(quarry_image,(xpos,ypos))
    background.save(name)
#TODO find a better way of doing this for example pass a quaternion in instead of new_corners
def skew(img,new_corners):
    width,height=img.size
    coeffs=find_coeffs([(0,0),(width,0),(width,height),(0,height)],new_corners)
    return img.transform(img.size,Image.PERSPECTIVE,coeffs,Image.BICUBIC)


def find_coeffs(current_plane,result_plane):
    matrix = []
    for p1, p2 in zip(result_plane, current_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(current_plane).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)
#no longer used may come back to
def rotateRandomly(quarry_image):
    degree=random.randint(0,360)
    return quarry_image.rotate(degree)

def pasteRandomly(quarry_image, background):
    x = random.randint(0, background.width-quarry_image.width)
    y = random.randint(0, background.height-quarry_image.height)
    background.paste(quarry_image, (x, y))


