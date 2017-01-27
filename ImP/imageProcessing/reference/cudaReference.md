# CUDA Reference
## PyCUDA
* https://developer.nvidia.com/how-to-cuda-python
* NVIDIA Supported Python Bindings: https://mathema.tician.de/software/pycuda/
   * https://github.com/inducer/pycuda (last commit Jan. 14, 2017 as of Jan. 25, 2017)
* Requirements
   * OpenCV
   * numpy
* Installation (as seen [here](https://github.com/inducer/pycuda/blob/master/README_SETUP.txt))
```
git clone https://github.com/inducer/pycuda.git
cd pycuda
sudo su
python3 setup.py build
python3 setup.py install
```
   * If nvcc is not on path, may need to edit ```/etc/environment``` with location to cuda binaries (e.g., ```/usr/local/cuda/bin```)
* [Tutorial](https://documen.tician.de/pycuda/tutorial.html)

## Using CUDA
* http://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s
* http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf