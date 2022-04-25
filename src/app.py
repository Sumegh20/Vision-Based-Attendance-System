import argparse
import tkinter as tk
import logging as lg

from src.collect_traning_data.get_face_images_from_camera import TrainingDataCollector
from src.face_embedding.faces_embedding import GenerateFaceEmbedding
from src.training.train_model import TrainFaceRecognitionModel
from src.predictor.facePredictor import FacePredictor

# create logger
lg.basicConfig(filename='logfile/Applicationlog.log',
               level=lg.INFO,
               format = '%(asctime)s: %(name)-2s : %(levelname)2s : %(message)s',
               datefmt='%d-%m-%y %H:%M')


class Application(tk.Tk):
    # __init__ function for class tkinterApp
    def __init__(self, *args, **kwargs):
        try:
            # __init__ function for class Tk
            tk.Tk.__init__(self, *args, **kwargs)

            # creating a container
            container = tk.Frame(self)
            container.pack(side="top", fill="both", expand=True)
            container.configure(background='#232F34')
            container.grid_rowconfigure(0, minsize=500)
            container.grid_columnconfigure(0, minsize=700)

            # initializing frames to an empty array
            self.frames = {}

            # iterating through a tuple consisting of the different page layouts
            for F in (Home, Registration):
                frame = F(container, self)

                # initializing frame of that object from Home, Registration respectively with for loop
                self.frames[F] = frame

                frame.grid(row=0, column=0, sticky="nsew")

            self.show_frame(Home)
        except Exception as e:
            lg.error("(__init__): Something went wrong in __init__ function of Application class\n \t"+str(e))
            raise Exception("(__init__): Something went wrong in __init__ function of Application class\n" + str(e))

    # to display the current frame passed as parameter
    def show_frame(self, cont):
        try:
            frame = self.frames[cont]
            frame.tkraise()
        except Exception as e:
            lg.error("(show_frame): Something went wrong in show_frame of Application class\n \t"+str(e))
            raise Exception("(show_frame): Something went wrong in show_frame of Application class\n \t"+str(e))


# first window frame Home
class Home(tk.Frame):
    def __init__(self, parent, controller):
        try:
            tk.Frame.__init__(self, parent)

            # label of frame Layout 2
            header = tk.Label(self, text="Attendance Module", width=80, height=2, fg="white",
                              bg="#344955", font=('times', 11, 'bold'))
            header.place(x=0, y=0)

            self.message = tk.Label(self, text="", bg="#bbc7d4", fg="black", width=61, height=2,
                                    activebackground="#bbc7d4", font=('times', 15))
            self.message.place(x=10, y=150)

            self.message.configure(text="If you are a new user then click Register Now")

            predictImg = tk.Button(self, text="Give Your Attendance", command=self.predict_images, fg="white",
                                   bg="#344955",
                                   width=18, height=1, activebackground="#607D8B", font=('times', 15, ' bold '))
            predictImg.place(x=50, y=270)

            registrationButton = tk.Button(self, text="Register Now", command=lambda:controller.show_frame(Registration),
                                           fg="white", bg="#344955", width=10, height=1, activebackground="#607D8B",
                                           font=('times', 15, ' bold'))
            registrationButton.place(x=500, y=270)

            auther = tk.Label(self, text="@Sumegh20", fg="blue", )
            auther.place(x=620, y=470)
        except Exception as e:
            lg.error("(__init__): Something went wrong in __init__ function of Home class\n \t" + str(e))
            raise Exception("(__init__): Something went wrong in __init__ of Home class\n \t" + str(e))

    def predict_images(self):
        try:
            faceDetector = FacePredictor()
            faceDetector.detectFace()
            notificationMessage = "Your attendance recorded successfully in the system."
            self.message.configure(text=notificationMessage)
        except Exception as e:
            lg.error("(predict_images): Something went wrong in predict_images of Home class\n \t" + str(e))
            raise Exception("(predict_images): Something went wrong in predict_images of Home class\n \t" + str(e))


# second window frame Registration
class Registration(tk.Frame):

    def __init__(self, parent, controller):
        try:
            tk.Frame.__init__(self, parent)

            header = tk.Label(self, text="Employee Monitoring Registration", width=80, height=2, fg="white",
                              bg="#344955", font=('times', 11, 'bold'))
            header.place(x=0, y=0)

            name_label = tk.Label(self, text="Employ Name", font=('bold', 14), fg="white", bg="#344955")
            name_label.place(x=10, y=80)
            self.name_label = tk.Entry(self, width=20, font=('bold', 14), bg="#bbc7d4", fg="black")
            self.name_label.place(x=140, y=80)

            self.message = tk.Label(self, text="", bg="#bbc7d4", fg="black", width=61, height=2,
                                    activebackground="#bbc7d4", font=('times', 15))
            self.message.place(x=10, y=150)
            self.message.configure(text="Enter your name first. Then click on Take Images button. It takes 50 images")

            takeImg = tk.Button(self, text="Take Images", command=self.input_images, fg="white",
                                bg="#344955", width=10, height=1, activebackground="#607D8B", font=('times', 15, ' bold'))
            takeImg.place(x=150, y=280)

            trainImg = tk.Button(self, text="Train Images", command=self.train_images, fg="white", bg="#344955",
                                 width=10, height=1, activebackground="#607D8B", font=('times', 15, ' bold'))
            trainImg.place(x=420, y=280)

            finish = tk.Button(self, text="Finish", command=lambda: controller.show_frame(Home), fg="white",
                               bg="#344955", width=10, height=1, activebackground="#607D8B", font=('times', 15, 'bold'))
            finish.place(x=560, y=370)

            auther = tk.Label(self, text="@Sumegh20", fg="blue", )
            auther.place(x=620, y=470)

            self.isRegister = False
        except Exception as e:
            lg.error("(__init__): Something went wrong in __init__ of Registration class\n \t" + str(e))
            raise Exception("(__init__): Something went wrong in __init__ of Registration class\n \t" + str(e))


    def input_images(self):
        try:
            name = (self.name_label.get())
            if len(name) == 0:
                notificationMessage = "Please, Enter your name first !!!"
                self.message.configure(text=notificationMessage)
            else:
                ap = argparse.ArgumentParser()
                ap.add_argument("--faces", default=50, help="Number of faces that camera will get")
                ap.add_argument("--output", default="../datasets/train/" + name, help="Path where the faces are store")

                args = vars(ap.parse_args())

                take_image_obj = TrainingDataCollector(args)
                take_image_obj.collectImagesFromCamera()

                notificationMessage = "We have collected " + str(args["faces"]) + " images for training."
                self.message.configure(text=notificationMessage)

                self.isRegister = True
        except Exception as e:
            lg.error("(input_images): Something went wrong in input_images of Registration class\n \t" + str(e))
            raise Exception("(input_images): Something went wrong in input_images of Registration class\n \t" + str(e))

    def getFaceEmedding(self):
        try:
            ap = argparse.ArgumentParser()
            ap.add_argument("--datasets", default="../datasets/train", help="Path to training dataset")
            ap.add_argument("--embeddings", default="faceEmbeddingModel/embeddings.pickle",
                            help="Path where the embedding file is store")

            args = vars(ap.parse_args())

            create_embedding = GenerateFaceEmbedding(args)
            create_embedding.getFaceEmbedding()
        except Exception as e:
            lg.error("(getFaceEmedding): Something went wrong in getFaceEmedding of Registration class\n \t" + str(e))
            raise Exception("(getFaceEmedding): Something went wrong in getFaceEmedding of Registration class\n \t" + str(e))


    def train_images(self):
        try:
            if self.isRegister == True:

                ap = argparse.ArgumentParser()
                ap.add_argument("--embeddings", default="faceEmbeddingModel/embeddings.pickle", help="Path where the embedding file is store")
                ap.add_argument("--model", default="faceEmbeddingModel/my_model.h5", help="path where the train model will store")
                ap.add_argument("--labelEncoder", default="faceEmbeddingModel/label_encoder.pickle", help="path where label encoder file is save")

                args = vars(ap.parse_args())

                self.getFaceEmedding()

                faceRecogModel = TrainFaceRecognitionModel(args)
                faceRecogModel.trainKerasModelForFaceRecognition()

                notificationMessage = "Model training is successful. Click on Finish Button"
                self.message.configure(text=notificationMessage)
            else:
                notificationMessage = "Please, Register yourself first !!!"
                self.message.configure(text=notificationMessage)
        except Exception as e:
            lg.error("(train_images): Something went wrong in train_images of Registration class\n \t" + str(e))
            raise Exception("(train_images): Something went wrong in train_images of Registration class\n \t" + str(e))

# Driver Code
app = Application()
app.mainloop()
