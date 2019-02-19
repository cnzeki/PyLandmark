# PyLandmark

![](https://github.com/cnzeki/PyLandmark/blob/master/demo/demo.jpg)

A facial landmark detector and pose estimator for python.

- The landmark detector is based on [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace).If you use `MTCNN` face detector,  the 5 facial landmarks can be used to get a better initial pose, which will be much faster than the original code.
- The pose estimator is from [sdm](https://github.com/chengzhengxin/sdm). 

[pyboostcvconverter](https://github.com/Algomorph/pyboostcvconverter) is used for data structure convertion between cv::Mat and numpy.

The code depends **on `OpenCV`**, other depends like ~~`OpenBlas`~~ and ~~`TBB`~~ were removed to  make it easier to compile.

## Depends

- CMake
- Opencv
- numpy


## Setup
```shell
git clone https://github.com/cnzeki/PyLandmark
cd PyLandmark
python setup.py build
python setup.py install

echo PyLandmark model dir: `pwd`/model
```

## Usage

```
# Init specify model dir
LandmarkDetector.create("./model/")

# Detect 68 landmarks
# rect:[x,y,w,h]
# point5:[x1,y2,..,x5,y5] or []
# type: 0 if point5 else 1
# return:[x1,y2,..,x68,y68]
pts = LandmarkDetector.detect(im, rect, point5, 0)

# Pose estimate
# return [pitch, yaw, roll] in degree
pose = LandmarkDetector.getPose(pts)

# Cleanup
LandmarkDetector.destroy()
```

## Demo

Since we have to do face detection first, check out [MtcnnPyCaffe](https://github.com/cnzeki/face-datasets) alongside this project.

```
python demo/demo.py demo/1.jpg
...
python demo/demo.py demo/4.jpg
```

Check the result images in the `demo` directory.

