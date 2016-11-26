# Fiducial Marker Module
## Purpose
Provides functionality to
1. Create fiducial markers
2. Identify fiducial markers by providing the data used to encode the markers

## Python Bindings
Aruco (as are other OpenCV modules) is written in C++. However, the automated process used by OpenCV to map C++ function calls and variables is implemented for Aruco in the latest GitHub repositories of [opencv](https://github.com/opencv/opencv) and [opencv_contrib](https://github.com/opencv/opencv_contrib/).

Python bindings for the Aruco module can be found by executing the following commands.
```
>>> python3
>>> from cv2 import aruco
>>> help(aruco)
```

## Markers and Dictionaries
Reference: [OpenCV Tutorial on Aruco Detection](http://docs.opencv.org/trunk/d5/dae/tutorial_aruco_detection.html)

### Introduction
An Aruco **marker** is a square represented by a set number of bits (e.g., a 4x4 marker uses 16 bits). If given the set that the marker belongs to, a marker's ID can be uniquely identified by decoding the bit representation.

An Aruco **dictionary** is a set of unique markers. Each dictionary is defined with the following characteristics:
* **Marker size** - the number of bits utilized by the marker (e.g., a 4x4 marker uses 16 bits)
* **Number of markers** - how many unique markers are present in the dictionary

For instance, aruco.DICT_6X6_250 is a dictionary with 6x6 bit markers (36 bits) and 250 uniquely identifiable markers.

Markers inside of a dictionary are referenced by an ID. For example, aruco.DICT_6X6_250 would have markers with ID in the inclusive range of [0,249].

### Choosing a Dictionary
When choosing a dictionary, one must consider three different properties.
1. Marker size
2. Number of markers in dictionary
3. Inter-marker distance

The **inter-marker distance** is the minimum distance among the markers and determines error detection/correction capabilities. Lower dictionary sizes and higher marker sizes increase the inter-marker distance and make it more robust to error.

Therefore, it is helpful to choose a dictionary with only as many markers as needed.

### Creating a Marker
The function signature for aruco.drawMarker(...) is defined as:
```
drawMarker(...)
    drawMarker(dictionary, id, sidePixels[, img[, borderBits]]) -> img
```
* **dictionary** - the dictionary object used to draw the marker
* **id** - the marker ID (e.g., for aruco.DICT_6X6_250, valid marker IDs are in the range of 0 through 249)
* **sidePixels** - length of each side of the output marker image (e.g., a value of 200 will result in a 200x200 pixel image); the value cannot be less than the number of bits required for each size (e.g., 5 is not a valid argument for a marker from a 4x4 dictionary, which requires 4 bits per side for the marker and 2 additional bits for the border), and the value should be proportional to the number of bits + border size OR much higher than the marker size to avoid deformations
* **img** - optional image output argument, since the marker image is already given as the return value
* **borderBits** - optional parameter to specify the width of the marker's black border, proportional to the size of the internal bits (e.g., a value of 2 means a border width twice the width of an internal bit)
