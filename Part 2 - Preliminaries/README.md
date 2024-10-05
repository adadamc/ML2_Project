# CSCI 4052U: Final Project
# Part 2: Preliminaries

## Requirement 1: Collect data
- Downloaded 20 GB worth of data (100,000+ Images and 5.4M+ Region Descriptions)
- Modified tutorial to work with local version instead of API version (which is no longer supported)
- Looked into how Visual Genome describes different regions of various images
- Many regions/descriptions per image, in good detail would be useful for this Project
- Very large dataset (may have to narrow it down for project later)

**To see more of what I did for collecting data/exploring the Visual Genome dataset: [LINK](visualgenometest.ipynb)**


## Requirement 2: Use simple model architectures to make preliminary attempts on some parts of the overall ML pipeline
- Tested out two different pre-trained models (alexnet and resnet152)
- Explored how each model handles various different photos
- Determined resnet152 does better and detects smaller details (at the cost of more processing time)
- Will attempt to use resnet152 or similar for project based on better results, but may need to use a simpler model or trimmed down dataset due to complexity

**To see more of what I did for exploring these pre-trained models: [LINK](pillow_notebook.ipynb)**


## Other Notes
#### Downloaded the following:
- OpenCV: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- matplotlib
- PyQt6
- torchvision
- visualgenome (https://github.com/ranjaykrishna/visual_genome_python_driver/tree/master)

**NOTE: Visual Genome contains ~20 GB worth of images and data, so I have excluded them from the cloud repository**

```
pip install jupyterlab
pip install opencv-contrib-python
pip install opencv-python
pip install matplotlib
pip install PyQt6
pip install torchvision
pip install . (inside of visualgenome folder)
```

#### Tutorials / Resources Used
- Object Detection: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
- Object Detection (using pre-trained network w/ TorchVision and Pillow): https://learnopencv.com/pytorch-for-beginners-image-classification-using-pre-trained-models/
- Visual Genome (https://homes.cs.washington.edu/~ranjay/visualgenome/api_beginners_tutorial.html) - modified to work with local version as API version no longer works

#### Credit
- Visual Genome https://homes.cs.washington.edu/~ranjay/visualgenome/index.html