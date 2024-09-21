# CSCI 4052u Project Proposal

## Project Idea
Analyze photos to detect objects within the photos and provide a detailed description, as well as answering questions about the photo.

## Purpose
To help the visually impaired by providing a text description (that could be read out) of what the scene looks like. I imagine this (if executed well) could be used in smart glasses or a similar device to help the visually impaired understanding their surroundings.

## Elements

### Vision
Analyzing the uploaded photo

### Natural Language
Describing the photo in words and potentially answering follow-up questions

### Reinforcement Learning (if time permits)
Potentially try and find a safe path to walk... for example when approaching a crosswalk this would be able to guide the user to not accidentally enter the road.

## Encoding the Domain and Range

**Function (initial upload):** Image -> Function -> String (Description of Image) and/or List of Strings (All objects found in image)

**Function (potential follow-up questions about image):** String A -> Function -> String B

### Describe how you plan to encode the domain and range of the task as vectors, matrices and tensors.

Source: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

For image classification, using Pillow/OpenCV may be useful for converting image data to a matrix (or three for coloured images: R,G,B) showing each pixel's value (source: https://www.geeksforgeeks.org/opencv-overview/).

For NLP, I intend to use a tokenizer/vocab system similar to what was used in CSCI4050 (https://csci4050u.science.ontariotechu.ca/part-5/5-lstm.html)



## Data Collection

Possible datasets for Computer Vision and captioning (some recommendations provided by Gemini):

- https://cocodataset.org/#home (COCO is a large-scale object detection, segmentation, and captioning dataset.)

 - https://paperswithcode.com/dataset/flickr30k (The Flickr30k dataset contains 31,000 images collected from Flickr, together with 5 reference sentences provided by human annotators.)

 - https://homes.cs.washington.edu/~ranjay/visualgenome/index.html (Offers detailed annotations about images, ex. an image may have an annotation saying there is a sign and that it is describing a bakery)
