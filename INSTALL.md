# Installation

## Requirements
- Python >= 3.6
- Numpy
- PyTorch >= 1.3 and matching torchvision, CUDA >= 10.0
```
    # check `nvcc --version` for CUDA version
    # Assuming Linux with anaconda - 

    # if CUDA == 10.0
    conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.0 -c pytorch
    
    # if CUDA == 10.1
    conda install pytorch=1.3.1 torchvision=0.4.2 cudatoolkit=10.1 -c pytorch
``` 
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- simplejson: `pip install simplejson`
- GCC >= 4.9
- OpenCV: build from source w/ ffmpeg support (preferred). Last resort: `pip install opencv-python`
- [Detectron2](https://github.com/facebookresearch/detectron2): 
```
    pip install -U cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
    git clone https://github.com/facebookresearch/detectron2 detectron2
    pip install -e detectron2
    # You could find more details at https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md
```

## PySlowFast

Clone the PySlowFast Video Understanding repository.
```
git clone git@github.com:shapes-a1/SlowFast.git
```

Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH

# or add to .bashrc or .zshrc
```

### Build PySlowFast

After having the above dependencies, run:
```
cd SlowFast
python setup.py build develop
```

<!--- Now the installation is finished, run the SlowFast (only) pipeline with:
```
# replace VIDEO_DEMO.DATA_SOURCE and VIDEO_DEMO.LABEL_FILE_PATH in cfg file with appropriate paths
python tools/run_net.py --cfg video_cfg/SLOWFAST_64x2_R101_50_50.yaml
```
-->
