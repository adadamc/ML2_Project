# Part 4: NLP

## Models Used

I used 2 pre-trained models for NLP. For the project the plan with regard to NLP is to describe an image (caption) and answer questions provided by the user. The two models below will be used to complete these goals. I plan to integrate NLP with the CV portion later to provide as much useful detail about an image as possible.
<br />
NLP Test (bottom of notebook): https://github.com/adadamc/ML2_Project/blob/main/Part%204%20-%20NLP/NLP_Part4.ipynb

### ViLT
(Visual Question Answering) <br />
Link: https://huggingface.co/dandelin/vilt-b32-finetuned-vqa <br />
Paper: https://arxiv.org/abs/2102.03334 <br />
ViLT is a pre-trained model on the VQAv2 dataset. <br />
Testing Results: Answers appear reasonable, questions like "What time of day is it?" give the answer "night". The question "What colour is the mailbox?" provides the correct answer "red". It is also able to answer how many traffic intersections are visible in an image. The question "What type of vehicle is prominent in the image?" correctly returns train for a streetcar that takes up most of the image. It was also able to answer that the streetcar has a green traffic light.

### BLIP
(Image Captioning) <br />
Link: https://huggingface.co/Salesforce/blip-image-captioning-large <br />
Paper: https://arxiv.org/abs/2201.12086 <br />
BLIP is a Visual Language Processing (VLP) framework. <br />
Testing Results: Returns reasonable captions (I do not expect it to return extremely detailed results, I'll see if I can incorporate some of the CV objects detected in the final project to compliment the caption). Output examples (where "a photo of" is provided by me as a starting point): "a photo of a city street with tall buildings and a tall tower
people walking on a sidewalk in a city at night" and "a photo of a city street with a train passing by at night
araffled city street with a red and white train passing by at night". The examples provide a good level of detail, while not always perfect detail, it is able to explain the main gist of the image.
          
