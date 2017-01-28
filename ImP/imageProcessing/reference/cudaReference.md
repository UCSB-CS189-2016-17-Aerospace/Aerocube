# CUDA Reference
## Concepts of CUDA
* **Host** - the CPU; code run on the CPU is the *host code*
* **Device** - refers to the GPU; code on the GPU is *device code*
* **CUBIN** - CUDA binary; ELF-formatted file that contains CUDA executable code sections
### References
* [An Easy Introduction to CUDA](https://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/)
* [An Even Easier Introduction to CUDA](https://devblogs.nvidia.com/parallelforall/even-easier-introduction-cuda/)
   * This was literally posted two days ago, on Jan. 25!!!
* https://mathema.tician.de/dl/main.pdf
* http://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s
* http://www.nvidia.com/docs/IO/116711/sc11-cuda-c-basics.pdf
## PyCUDA
* Overview
   * Allows creation of CUBIN code (called modules) embedded into Python code
   * [Tutorial](https://documen.tician.de/pycuda/tutorial.html)
   * https://developer.nvidia.com/how-to-cuda-python
    * NVIDIA Supported Python Bindings: https://mathema.tician.de/software/pycuda/
       * https://github.com/inducer/pycuda (last commit Jan. 14, 2017 as of Jan. 25, 2017)
* Requirements
   * OpenCV
   * numpy
* Installation (as seen [here](https://github.com/inducer/pycuda/blob/master/README_SETUP.txt))
   * If nvcc is not on path, may need to edit ```/etc/environment``` with location to cuda binaries (e.g., ```/usr/local/cuda/bin```)
```
git clone https://github.com/inducer/pycuda.git
cd pycuda
sudo su
python3 setup.py build
python3 setup.py install
```


## Using CUDA
