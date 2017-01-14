from PIL import Image, ImageFilter

background = Image.new("RGB",(1000,1000))
img = Image.open('thing.png')
width, height = img.size
img=img.transform(img.size,Image.PERSPECTIVE,(1,1,0, 0, 1, 0, 0, 0))
background.paste(img,(0,0))
background.filter(ImageFilter.GaussianBlur(10))
background.save('test.png')