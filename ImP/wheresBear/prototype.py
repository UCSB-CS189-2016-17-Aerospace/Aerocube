from PIL import Image

background = Image.new("RGB",(1000,1000))
object = Image.open('thing.png')
background.paste(object,(0,0))
background.save('test.png')