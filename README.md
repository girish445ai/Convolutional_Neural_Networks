# Convolutional_Neural_Networks

## PART B: Fine-tuning a pre-trained model:

* ### How to run the code: 

* Step 1: 

Run all the code until  pretrain function and then the sweep command to observe the sweeps . 

* Step2: 

For evaluating the best model  on the test dataset, DONOT run the pretrain function and sweep command and go to the Optimal pre train function and run the code from that function . 

## PART C:Using a pre-trained model as it is:
* The pre-trained Yolov4 model has been used for object identification on a video. YOLO stands for ‘you only look once’ and YOLOv4 is the 4th addition to the family of YOLO object detector models. It is a milestone model which solidified YOLO’s name and position in the computer vision field.The Yolov4 has the higher frame rate when compared to its previous models . It also shows better accuracy rates .
* For the purpose of the YOLOv4 object detection, we will be making use of its pre-trained model weights on Google Colab. The pre-trained model was trained on the MS-COCO dataset which is a dataset of 80 classes engulfing day-to-day objects. This dataset is widely used to establish a benchmark for the purposes of detection and classification. 
* ### How to run the code:
* Run all the cells in notebook . The steps include:
 1) Cloning the Github repository of the official Darknet YOLOv4 architecture.YOLOv4 uses Darknet which is a neural network framework written in C and CUDA.
 2) Run the make command that builds darknet i.e. convert the darknet code into an executable file/application.
 3) Download Pre-trained YOLOv4 Model Weights using wget command.
 4) Perform object detection in videos using Darknet CLI command :
 5) !./darknet detector demo cfg/coco.data cfg/yolov4.cfg yolov4.weights -dont_show (Video path) -i 0 -out_filename results.avi.
 6)The video path is path of video file on which we want to perform object detection . It should be uploaded to drive and path should be given in the command.

* Below is the YouTube link of our video. We have used 2 videos, both are from YouTube ,one is the video of a runway bull crashing through the people on road and the other is the videoclip from junglebook where Mowgli is surrounded by animals .

* Link to object detection performed on bullcrashing video:
https://youtu.be/ufAQ8zx8MSE

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/ufAQ8zx8MSE/0.jpg)](https://www.youtube.com/watch?v=ufAQ8zx8MSE)

* Link to object detection performed on Junglebook video :
https://youtu.be/rtucVWcPhr8

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/rtucVWcPhr8/0.jpg)](https://www.youtube.com/watch?v=rtucVWcPhr8)

## Social Relevance :
* The social relevance of our project is that to detect animals in proximity to humans space ,the danger could be both ways that the humans could hurt animals or animals could hurt humans . We want to detect such activities and we can create a animal intrusion system where it alerts people when animals are casuing danger to humans.
*  The other way is that this system could also be installed in animal space like forests or zoos when humans try to hunt or cause harm to animals it could detect and alert .  
* Another Social relevance is that this system could alert when there is Crop damage caused by animal attacks . Crop damage is one of the major threats in reducing the crop yield. Due to the expansion of cultivated land into previous wildlife habitat, crop raiding is becoming one of the most antagonizing human-wildlife conflicts. Farmers in India face serious threats from pests, natural calamities & damage by animals resulting in lower yields. Traditional methods followed by farmers are not that effective and it is not feasible to hire guards to keep an eye on crops and prevent wild animals. Since safety of both human and animal is equally vital, it is important to protect the crops from damage caused by animal as well as divert the animal without any harm. We can use this system to monitor the entire farm at regular intervals through a camera and to do the necessary action .
