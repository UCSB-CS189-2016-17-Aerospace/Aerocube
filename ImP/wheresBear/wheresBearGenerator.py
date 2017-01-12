from PIL import Image
import random

def generateDatabase(size, quarry_images, backgrounds=[Image.new("RGB", (1000, 1000))]):
    for i in range(size):
        createImage(quarry_images[0], backgrounds[0], '%i' % (i))

def createImage(quarry_image, background, name):
    quarry_image=rotateRandomly(quarry_image)
    pasteRandomly(quarry_image, background)
    background.save(name)

def rotateRandomly(quarry_image):
    degree=random.randint(0,360)
    return quarry_image.rotate(degree)

def pasteRandomly(quarry_image, background):
    x = random.randint(0, background.width-quarry_image.width)
    y = random.randint(0, background.height-quarry_image.height)
    background.paste(quarry_image, (x, y))


