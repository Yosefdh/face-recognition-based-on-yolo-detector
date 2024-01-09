# final project in computer vision and machine learning #
import dlib
import numpy as np
import pandas as pd
import os, sys
import cv2
import torch
import torchvision
from dlib import face_recognition_model_v1
import face_recognition
import torchvision.transforms as T
from PIL import Image

class FaceRecognition:
    face_locations=[]
    face_encodings=[]
    face_names=[]
    confidence = []
    known_face_encodings=[]
    known_face_names=[]
    process_this_frame = True

    def __int__(self):
        self.encode_faces()
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image=face_recognition.load_image_file(f'faces/{image}')
            face_encoding=face_recognition.face_encodings(face_image,num_jitters=50)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)
    def face_detector(self,frame,model):
        #self.detector = torch.hub.load('yolov5-master', 'custom', source='local', path='yolov5-master/face3.pt')
        self.detector= model
        detections = self.detector(frame, size=640)
        detections = detections.pandas().xyxy[0].to_dict(orient="records")
        self.confidence = []
        self.locations = []
        self.cs = []

        for result in detections:
            confidence = result['confidence']
            cs = result['class']
            top = int(result['ymax'])
            right = int(result['xmax'])
            bottom = int(result['ymin'])
            left = int(result['xmin'])
            loc_list=(top,right,bottom,left)
            self.locations.append(np.array(loc_list))
            self.confidence.append(confidence)
            self.cs.append(cs)
        return (self.cs,self.locations,self.confidence)





    def run_recognition(self):
        model = torch.hub.load('/home/yosef/yosef/Final project/yolov5-master', 'custom', source='local', path='face3.pt')
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            sys.exit('Video source not found')

        while (True):
            ret, frame = cam.read()

           # frame = cv2.cvtColor(np.array(frame), cv2.COLOR_BGR2RGB)
            #frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
            if self.process_this_frame:
                # self.face_locations=face_recognition.face_locations(frame)
                self.face_locations = self.face_detector(frame,model)[1]
                self.face_encodings =face_recognition.face_encodings(frame,self.face_locations)
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings,face_encoding,tolerance=0.65)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings,face_encoding)
                    best_match_index = int(np.argmin(face_distances))
                    print(best_match_index)
                    if matches[best_match_index]:
                        name=self.known_face_names[best_match_index]
                    self.face_names.append((f'{name}({confidence})'))
            self.process_this_frame = not self.process_this_frame

            for(top,right,bottom,left), name in zip(self.face_locations,self.face_names):
                # top*=4
                # right*=4
                # bottom*=4
                # left*=4
                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                cv2.rectangle(frame, (left, bottom-35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame,name,(left+6,bottom-6),cv2.FONT_HERSHEY_DUPLEX,0.8,(255,255,255),1)
                cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cam.release()
        cv2.destroyAllWindows()

if __name__=='__main__':

    # detector = torch.hub.load('yolov5-master', 'custom', source='local', path='yolov5-master/face3.pt')
    fr = FaceRecognition()
    fr.encode_faces()
    fr.run_recognition()
