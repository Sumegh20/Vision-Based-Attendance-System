import sys
from src.insightface.src.common import face_preprocess

sys.path.append("../insightface/deploy")
sys.path.append("../insightface/src/common")

from mtcnn.mtcnn import MTCNN
import numpy as np
import cv2
import os
from datetime import datetime

class TrainingDataCollector:
    def __init__(self, args):
        self.args = args
        self.detector = MTCNN()

    def collectImagesFromCamera(self):
        try:
            # initialize video stream
            cap = cv2.VideoCapture(0)

            faces = 0
            frames = 0
            max_faces = int(self.args["faces"])
            max_bbox = np.zeros(4)

            # Create the path where the images will store
            if not (os.path.exists(self.args["output"])):
                os.makedirs(self.args["output"])

            while faces < max_faces:
                ret, frame = cap.read()
                frames += 1

                dtString = str(datetime.now().microsecond)
                bboxes = self.detector.detect_faces(frame)

                # Get only the biggest face
                if len(bboxes) != 0:
                    max_area = 0
                    for bbox_i in bboxes:
                        bbox = bbox_i["box"]
                        bbox = np.array([bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]])
                        keypoints = bbox_i["keypoints"]
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

                        if area > max_area:
                            max_area = area
                            max_bbox = bbox
                            landmarks = keypoints

                    max_bbox = max_bbox[0:4]

                    # get each of 3 frames
                    if frames % 3 == 0:
                        landmarks = np.array([landmarks["left_eye"][0], landmarks["right_eye"][0], landmarks["nose"][0],
                                             landmarks["mouth_left"][0], landmarks["mouth_right"][0],
                                             landmarks["left_eye"][1], landmarks["right_eye"][1], landmarks["nose"][1],
                                             landmarks["mouth_left"][1], landmarks["mouth_right"][1]])
                        landmarks = landmarks.reshape((2, 5)).T

                        # preprocessing the images
                        nimg = face_preprocess.preprocess(frame, max_bbox, landmarks, image_size='112,112')

                        cv2.imwrite(os.path.join(self.args["output"], "{}.jpg".format(dtString)), nimg)
                        cv2.rectangle(frame, (max_bbox[0], max_bbox[1]), (max_bbox[2], max_bbox[3]), (255, 0, 0), 2)
                        print("[INFO] {} Image Captured".format(faces + 1))
                        faces += 1

                cv2.imshow("Face detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()

        except Exception as e:
            raise Exception(f"(collectImagesFromCamera): Something went wrong on image collection\n" + str(e))