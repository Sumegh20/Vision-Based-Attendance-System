# Vision Based Attendance System 

Today there is a need for automation systems by which 
we can take attendance automatically. The face detection 
system has also delivered a security enhancement, where 
this can be used in the system to provide greater 
security. There is no need of manual interventions and 
the attendance is taken automatically and is stored in 
the file.


## Collect Images
We collect 50 images using OpenCV then detect the face
using mtcnn and crop the face images and store them
into a folder

## Generate Face Embedding 
We Generate the face embeddings and store them in a pickle
file with the class name using insightface.

## Train Images
We build a classifier using ANN pass the face embeddings.
Store the model in my_model.h5

## Prediction
We collect images using OpenCV and the detect the face
using mtcnn and crop the face. Generate the face 
embeddings then using the Train model we predict the face
and store the probability, we also calculate the cosine 
similarity between the new face embeddings and the
predicted class face embeddings. If both are crose the 
certain thersold. Then we take the attendance and store it in a file

## Technology Stack
* Programming Language :: python 3.6
* Deep Learning Framework :: mxnet, keras
* Desktop Application :: tkinter
* Image Processing :: OpenCV
* Face Detection Algoritham :: mtcnn
* Face recognition Algoritham :: arcface

## Importent Information

* [How to Run](https://github.com/Sumegh20/Vision-Based-Attendance-System/blob/main/How%20to%20Run.txt)
* **To run this project you need atleast two person's imges in the datasets folder.**
![Capture](https://user-images.githubusercontent.com/81466184/165136668-62417b39-1848-4256-b1ce-878227cd6f75.PNG)

## How to control the Desktop application

Step 1: Run the [app.py](https://github.com/Sumegh20/Vision-Based-Attendance-System/blob/main/src/app.py), The following interface in open

![Image 1](https://user-images.githubusercontent.com/81466184/165136073-3e9a657a-36fd-466a-9290-813c64ec8c0f.PNG)

Step 2: Click "Register Now" button and you get following interface

![Image 2](https://user-images.githubusercontent.com/81466184/165136217-5182984d-8b3c-4fd5-8973-fabce52a6dca.PNG)

Step 3: First enter your name. 

Step 4: Click "Take Images" button

![Image 3](https://user-images.githubusercontent.com/81466184/165136308-9593bf01-db6e-424d-8dbb-22480f867c36.PNG)

Step 5: Click "Train Images" button.

![Image 4](https://user-images.githubusercontent.com/81466184/165136422-d032e7b7-361b-4da5-ae40-ad805497730f.PNG)

Step 6: Click "Finish"

Step 7: Click "Give Your Attendance" button for your attendance

![Image 5](https://user-images.githubusercontent.com/81466184/165136535-a042eb8e-0c68-4a0e-8cc6-7a0a49b50e3e.PNG)

**If you registered your self ones then only click "Give Your Attendance" button for attendance**









