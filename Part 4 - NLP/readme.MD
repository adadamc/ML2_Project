# Part 3: CV

Note: Downloading the Jupyter Notebooks may be required to see all code, web previews seem to be cut off.

## Initial Attempt
Originally I tried to use the VisualGenome dataset but due to the very large size, API access no longer being supported, and very detailed classes making it difficult to get a large enough amount of samples for each class.

Initial Attempt to work with VisualGenome is in this Jupyter Notebook: https://github.com/adadamc/ML2_Project/blob/main/Part%203%20-%20Computer%20Vision/CV_Part3_Outdated.ipynb

## New Plan
Combining multiple pre-trained models for multiple datasets.

Testing here: https://github.com/adadamc/ML2_Project/blob/main/Part%203%20-%20Computer%20Vision/CV_Part3.ipynb

Outline:
- Enabled GPU
- Used Faster R-CNN model with a ResNet-50-FPN (https://pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html) with the COCO dataset to detect multiple objects in a single image. Sources used are listed in the Jupyter notebook.
    - Started by transforming image similarly to as done in part 2
        - Resized to 256x256
        - Crop image to 224x224 about the center
        - Convert image to Tensor
        - Normalize Image
    - Run through the model
    - Returns boxes in a tensor
        - ```'boxes': tensor([[1.2335e+03, 4.5777e+02, 1.3936e+03, 6.2880e+02],
          [4.0404e+02, 2.5434e+03, 4.8625e+02, 2.7359e+03],
          [2.6461e+03, 2.6778e+03, 3.2237e+03, 2.9350e+03],
          [6.5539e+02, 2.4871e+03, 7.0804e+02, 2.6381e+03],
          [1.1637e+03, 4.7825e+02, 1.2332e+03, 6.0446e+02],
          [1.1208e+03, 5.2595e+02, 1.1735e+03, 6.1862e+02],
          [1.1586e+03, 4.7738e+02, 1.2107e+03, 5.6960e+02],
          [1.1229e+03, 4.7991e+02, 1.2229e+03, 6.2248e+02],
          [2.6946e+03, 2.3503e+03, 2.7952e+03, 2.4738e+03],
          [3.6081e+02, 2.5142e+03, 4.1145e+02, 2.6651e+03],
          [1.5182e+03, 2.4237e+03, 1.5713e+03, 2.4887e+03],
          [1.1450e+03, 4.8211e+02, 1.1989e+03, 5.9101e+02],
          [4.2158e+02, 2.3149e+03, 4.8212e+02, 2.3724e+03],
          [3.5742e+02, 2.4630e+03, 4.0897e+02, 2.5960e+03],
          [1.3313e+03, 2.5674e+03, 1.4036e+03, 2.6097e+03],
          [3.8084e+03, 2.8193e+03, 4.0609e+03, 3.0510e+03],
          [2.1488e+03, 2.6538e+03, 2.3156e+03, 2.7382e+03],
          [4.7698e+00, 2.5226e+03, 1.1275e+02, 2.6127e+03],
          [3.9949e+03, 2.8227e+03, 4.0458e+03, 2.9006e+03],
          [3.7304e+02, 2.4800e+03, 4.1577e+02, 2.5822e+03],
          [3.9136e+03, 2.8901e+03, 4.0567e+03, 3.0465e+03],
          [2.7072e+03, 2.3520e+03, 2.7683e+03, 2.4155e+03],
          [3.5158e+02, 2.4759e+03, 3.9218e+02, 2.5794e+03],
          [3.3445e+03, 2.6188e+03, 3.7233e+03, 3.0623e+03],
          [4.9219e+02, 2.5271e+03, 5.3411e+02, 2.6034e+03],
          [2.9162e+03, 2.6998e+03, 2.9616e+03, 2.7519e+03],
          [1.4685e+03, 2.5692e+03, 1.4975e+03, 2.6298e+03],
          [3.3645e+03, 2.6893e+03, 3.6691e+03, 3.0485e+03],
          [3.7526e+03, 2.8503e+03, 3.9633e+03, 3.0435e+03],
          [2.8969e+03, 2.6987e+03, 2.9440e+03, 2.7574e+03],
          [2.8819e+03, 2.6606e+03, 3.0246e+03, 2.7040e+03],
          [2.7615e+02, 2.4041e+03, 3.1338e+02, 2.4570e+03],
          [4.5213e+02, 2.5015e+03, 4.9054e+02, 2.5805e+03],
          [5.5680e+02, 2.4873e+03, 6.2149e+02, 2.6835e+03],
          [3.8333e+03, 2.9069e+03, 4.0086e+03, 3.0515e+03],
          [9.7048e+02, 2.4309e+03, 9.9513e+02, 2.4855e+03],
          [3.9461e+03, 2.8914e+03, 4.0722e+03, 3.0467e+03],
          [3.5648e+02, 2.5217e+03, 3.9567e+02, 2.6340e+03],
          [3.9781e+03, 2.7423e+03, 4.0800e+03, 3.0459e+03],
          [5.2129e+02, 2.4296e+03, 6.2949e+02, 2.7505e+03],
          [2.4249e+02, 2.3349e+03, 3.0111e+02, 2.3810e+03],
          [2.9101e+02, 2.3982e+03, 3.2386e+02, 2.4489e+03],
          [3.9691e+03, 2.8291e+03, 4.0382e+03, 2.8904e+03],
          [1.5183e+03, 2.4245e+03, 1.5708e+03, 2.4875e+03],
          [2.3906e+03, 2.6170e+03, 2.4242e+03, 2.6811e+03],
          [4.7175e+02, 2.5106e+03, 5.0346e+02, 2.5880e+03],
          [1.4827e+03, 2.5721e+03, 1.5110e+03, 2.6302e+03],
          [3.7640e+00, 2.5062e+03, 1.2564e+02, 2.6116e+03]])
    - Returns labels in a tensor (position in .txt document):
        **tensor([85,  1,  3,  1, 85, 85, 85, 85, 10,  1, 13, 85, 10,  1,  3,  2,3,  3, 1,  1,  2, 10,  1,  8,  1,  1,  1, 84,  2,  1,  3, 10,  1,  1,  2,  1,
          62,  1,  1,  1, 10, 10,  1, 85, 14,  1,  1,  8])**
        - Corresponding labels:
        ```
        ["person" , "bicycle" , "car" , "motorcycle" , "airplane" , "bus" , "train" , "truck" , "boat" , "traffic light" , "fire hydrant" , "street sign" , "stop sign" , "parking meter" , "bench" , "bird" , "cat" , "dog" , "horse" , "sheep" , "cow" , "elephant" , "bear" , "zebra" , "giraffe" , "hat" , "backpack" , "umbrella" , "shoe" , "eye glasses" , "handbag" , "tie" , "suitcase" , 
        "frisbee" , "skis" , "snowboard" , "sports ball" , "kite" , "baseball bat" , 
        "baseball glove" , "skateboard" , "surfboard" , "tennis racket" , "bottle" , 
        "plate" , "wine glass" , "cup" , "fork" , "knife" , "spoon" , "bowl" , 
        "banana" , "apple" , "sandwich" , "orange" , "broccoli" , "carrot" , "hot dog" , "pizza" , "donut" , "cake" , "chair" , "couch" , "potted plant" , "bed" ,"mirror" , "dining table" , "window" , "desk" , "toilet" , "door" ,"tv" ,
        "laptop" , "mouse" , "remote" , "keyboard" , "cell phone" , "microwave" ,
        "oven" , "toaster" , "sink" , "refrigerator" , "blender" , "book" ,
        "clock" , "vase" , "scissors" , "teddy bear" , "hair drier" , "toothbrush" , "hair brush"] 
    - Returns scores for each bounding box in a tensor (confidence): **'scores': tensor([0.9998, 0.9980, 0.9948, 0.9907, 0.9668, 0.9404, 0.7987, 0.6263, 0.3838,
          0.3663, 0.3624, 0.3573, 0.3341, 0.3329, 0.2941, 0.2697, 0.2688, 0.2593,
          0.2210, 0.2141, 0.1893, 0.1596, 0.1513, 0.1434, 0.1281, 0.1030, 0.1026,
          0.0998, 0.0923, 0.0912, 0.0908, 0.0867, 0.0809, 0.0786, 0.0752, 0.0733,
          0.0724, 0.0671, 0.0633, 0.0632, 0.0629, 0.0612, 0.0605, 0.0578, 0.0526,
          0.0518, 0.0506, 0.0503])**
    - For each bounding box where the confidence of the label is above 90%, I then added a rectangle displaying it on the image with the class above the rectangle and the confidence on the right of the rectangle.
    - I then also tested lower confidences, by including any bounding boxes with confidences above 60%, which showed fairly accurate results in all three confidence ranges, but only 90% above seemed to have no errors. A traffic/direction **sign** appears to have been mis-categorized as a traffic **light** at 80.5% confidence and what appears to be a red light on a building in the distance was mis-categorized as a traffic light with 66% confidence.
        - Green bounding box = 90%+
        - Yellow bounding box = 75-90%
        - Red bounding box = 60-75%
        - Anything below 60% is not shown
    - Also tested a blurry image (moving train and blurred humans), it worked reasonably well just with lower confidences.
    - Finally, I also tested Places365 with the VGG16 CNN pre-trained (https://github.com/GKalliatakis/Keras-VGG16-places365) using Keras. The models are conversions of models originally used Caffe instead of Keras (https://github.com/CSAILVision/places365). The Caffe version shows that out of AlexNet, GoogLeNet, VGG and ResNet: VGG performed the best in the highest confidence result being correct (~55%), and second best at guessing the correct class within the top 5 confidence guesses (~85%). Since the accuracy of top-1 guess is not the best (for any of the CNNs), I have decided to keep the top-5s for now and will later decide how to best incorporate them (perhaps by looking at how the results from COCO and Places365 relate).
    - The overall goal at this point when NLP is implemented is to give a result saying what the overall theme is, ex. city intersection, forest, school campus. Then it will also (ideally in detail) describe what it sees, ex. a bear is behind a tree in a forest.
    
          
