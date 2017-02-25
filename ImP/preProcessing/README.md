#PreProcessor 
A class with only one argument which is the path to a recently taken image. The class can perform various actions with the image such as check for similarities between itself and previous images, alter the contrast, etc.

## Checking for Image Similarity
We check for image similarity with the function **is_similar(self, pathToOtherImage)**. It simply takes in the path for the image you 
want to compare it to and performs 'subtraction' between the two images. This is done throgh the OpenCV method [cv2.subtract()](http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#void subtract(InputArray src1, InputArray src2, OutputArray dst, InputArray mask, int dtype)).
We then calculate the mean of the arrays that are returned to essentially quantify the difference between two images in just one number, with 0.0 representing two identical images.

Currently this threshold is set at 2.2. This values is tentative as the team will discuss what constitutes an image as 'too similar' to a previous one.
To provide a visual aid, the set of images below achieved a difference value of 2.18731918012. 


<img src="https://raw.githubusercontent.com/UCSB-CS189-2016-17-Aerospace/Aerocube/feature/similar_image/ImP/preProcessing/test_pictures/im1.JPG" width="600" length="600"/>

<img src="https://raw.githubusercontent.com/UCSB-CS189-2016-17-Aerospace/Aerocube/feature/similar_image/ImP/preProcessing/test_pictures/im4.JPG" width="600" length="600"/>

## gopro_calibration_images
The images in this directory are to be used to identify the calibration matrix and distorition matrix for the gopro camera.

## test_pictures
These images are test images that were used to calculate image differences, but may later be used as actual test images for the entire program.
