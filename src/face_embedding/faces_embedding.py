import sys

from src.insightface.deploy import face_model
sys.path.append("../insightface/deploy")
sys.path.append("../insightface/src/common")


from imutils import paths
import numpy as np
import pickle
import cv2
import os


class GenerateFaceEmbedding:
    def __init__(self, args):
        try:
            self.args = args
            self.image_size = '112,112'
            self.model = "insightface/models/model-y1-test2/model,0"
            self.threshold = 1.24
            self.det = 0
        except Exception as e:
            raise Exception(f"(GenerateFaceEmbedding(__init__)): Something went wrong in __init__() of GenerateFaceEmbedding\n" + str(e))

    def getFaceEmbedding(self):
        try:
            # Grab all the paths of the training images
            print("[Info] quantifying faces...")
            imagePaths = list(paths.list_images(self.args["datasets"]))

            # Initialize the faces embedder
            embedding_model = face_model.FaceModel(self.image_size, self.model, self.threshold, self.det)

            # Initialize the list of names and facial embedding
            EmbddingList = []
            NameList = []

            # Initialize the total number of faces processed
            count = 0

            for (i, imagePath) in enumerate(imagePaths):
                print(f"[info] processing image {i+1}/{len(imagePaths)}")

                # Extract the person name from the image path
                name = imagePath.split(os.path.sep)[-2]

                # Read the image
                image = cv2.imread(imagePath)

                # Convert face from BGR to RGB color
                nimg = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                nimg = np.transpose(nimg, (2, 0, 1))

                # Get the face embedding vector
                face_embedding = embedding_model.get_feature(nimg)

                # Append the name and embedding vector in the crosponding list
                NameList.append(name)
                EmbddingList.append(face_embedding)
                count += 1

            print(count, "faces embedded")

            # save to output
            data = {"embeddings": EmbddingList, "names": NameList}
            f = open(self.args["embeddings"], "wb")
            f.write(pickle.dumps(data))
            f.close()
        except Exception as e:
            raise Exception(f"(getFaceEmbedding): Something went wrong in getFaceEmbedding\n" + str(e))