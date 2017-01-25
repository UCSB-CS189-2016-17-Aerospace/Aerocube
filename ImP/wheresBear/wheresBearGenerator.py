import numpy
from PIL import Image, ImageColor
import os
import random

IMAGESIZE = 500
wheres_bear_dir = os.path.dirname(__file__)
# TODO: get rid of arbitrary constants


def generateDatabase(size, quarry_images, backgrounds=["background.png"]):
    # TODO: finish function
    for i in range(size):
        quarry_image_name = quarry_images[random.randint(0, len(quarry_images)-1)]
        background = backgrounds[random.randint(0, len(quarry_images)-1)]
        rotation_degree = random.randint(0, 360)
        size_of_quarry_image = random.randint(15, 25)
        pos = (random.randint(0, IMAGESIZE-size_of_quarry_image*6), random.randint(0, IMAGESIZE-size_of_quarry_image*6))

        print(quarry_images)
        print(background)
        print(rotation_degree)
        print(pos)
        print(size_of_quarry_image)
        createImage(quarry_image_name=quarry_image_name, background_name=background, rotation_degree=rotation_degree,
                    pos=pos, skew_values=randomSkewValues(10), size=size_of_quarry_image)


def createImage(quarry_image_name, background_name, rotation_degree=0, pos=(0, 0), size=1, skew_values=None):
    """
    :param quarry_image_name: image of quarry
    :param background_name: background image
    :param rotation_degree:
    :param pos: where to place the quarry images' top left corner when pasted defaults top left corner
    :param size: size scalar
    :param skew_values: 8 long array of offsets for the corners of the quarry image
    :return:
    """
    # hard-code files to save to wheresbear/database
    name = os.path.join(wheres_bear_dir,
                        'database',
                        'Rot{0}_Pos{1}_Siz{2}_skew{3}'.format(rotation_degree, pos, size, skew_values),
                        quarry_image_name)
    quarry_image = Image.open(quarry_image_name)
    background = Image.open(background_name)
    quarry_image = quarry_image.resize((quarry_image.width*size, quarry_image.height*size))
    mask = Image.new("RGBA", quarry_image.size, ImageColor.getrgb("black"))  # TODO: get rid of the boarder stuff
    # quarry_image=quarry_image.convert("RGB")

    if skew_values is not None:
        width, height = quarry_image.size
        corners = [(skew_values[0], skew_values[1]),
                   (width+skew_values[2], skew_values[3]),
                   (width+skew_values[4], height+skew_values[5]),
                   (skew_values[6], height-skew_values[7])]
        quarry_image = skew(quarry_image, corners)
        mask = skew(mask, corners)  # boarder stuff

    quarry_image = quarry_image.rotate(rotation_degree, expand=True)
    mask = mask.rotate(rotation_degree, expand=True)  # boarder stuff

    background.paste(quarry_image, pos, mask)
    background.save(name)


# TODO: find a better way of doing this for example pass a quaternion in instead of new_corners
def randomSkewValues(amount):
    values=[]
    for i in range(8):
        values.append(random.randint(-amount, amount))
    print(values)
    return values


def skew(img, new_corners):
    width, height = img.size
    coeffs = find_coeffs([(0, 0), (width, 0), (width, height), (0, height)], new_corners)
    return img.transform(img.size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)


def find_coeffs(current_plane,result_plane):
    matrix = []
    for p1, p2 in zip(result_plane, current_plane):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = numpy.matrix(matrix, dtype=numpy.float)
    B = numpy.array(current_plane).reshape(8)

    res = numpy.dot(numpy.linalg.inv(A.T * A) * A.T, B)
    return numpy.array(res).reshape(8)


# no longer used may come back to
def rotateRandomly(quarry_image):
    degree=random.randint(0,360)
    return quarry_image.rotate(degree)

def pasteRandomly(quarry_image, background):
    x = random.randint(0, background.width-quarry_image.width)
    y = random.randint(0, background.height-quarry_image.height)
    background.paste(quarry_image, (x, y))

