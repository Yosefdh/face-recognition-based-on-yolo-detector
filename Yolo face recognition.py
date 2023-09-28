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
#from ultralytics.yolo.data.dataloaders.v5augmentations import letterbox
#
# path = 'dataset'
# detector = torch.hub.load('yolov5-master', 'custom', source='local', path='yolov5-master/face3.pt')
#
# cam = cv2.VideoCapture(0)
# img_size= (640, 640)
# # For each person, enter one numeric face id
# face_id = input('\n enter user id end press <return> ==>  ')
#
# print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# # Initialize individual sampling face count
# count = 0
#
# while(True):
#
#     ret, img = cam.read()
#     # img = cv2.flip(img, -1) # flip video image vertically
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     det1 = detector(gray, size=640)
#     #tensor_array.cpu().detach().numpy()
#     crop=det1.crop()
#     crop = crop['box'].detach().numpy()
#     # face = det1.pandas().xyxy[0]  # updates results.ims with boxes and labels
#     faces = crop
#
#
#     for (xmin,ymin,xmax,ymax) in faces:
#
#         cv2.rectangle(img, (xmin,ymin), (xmax,ymax), (255,0,0), 2)
#         count += 1
#
#         # Save the captured image into the datasets folder
#         cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[ymin:ymax,xmin:xmax])
#
#         cv2.imshow('image', img)
#
#     k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video
#     if k == 27:
#         break
#     elif count >= 30: # Take 30 face sample and stop video
#          break
#
# # Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# cam.release()
# cv2.destroyAllWindows()
#
#
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
#
# # function to get the images and label data
# def getImagesAndLabels(path):
#     imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
#     faceSamples=[]
#     ids = []
#     for imagePath in imagePaths:
#         PIL_img = Image.open(imagePath).convert('L') # grayscale
#         img_numpy = np.array(PIL_img,'uint8')
#         id = int(os.path.split(imagePath)[-1].split(".")[1])
#         faces = detector(img_numpy,size=640)
#
#         for (x,y,w,h) in faces:
#             faceSamples.append(img_numpy[y:y+h,x:x+w])
#             ids.append(id)
#     return faceSamples,ids
# print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
#
# faces,ids = getImagesAndLabels(path)
# recognizer.train(faces, np.array(ids))
# # Save the model into trainer/trainer.yml
# recognizer.write('trainer/trainer.yml')
# # Print the numer of faces trained and end program
# print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))



#model = torch.hub.load('ultralytics/yolov5', 'custom',path='yolov5-master/yolov5-master/runs/train/exp20/weights/best.pt',force_reload='True')

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
            face_encoding=face_recognition.face_encodings(face_image)[0]

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
        model = torch.hub.load('yolov5-master', 'custom', source='local', path='yolov5-master/face3.pt')
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
                    matches = face_recognition.compare_faces(self.known_face_encodings,face_encoding)
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

    detector = torch.hub.load('yolov5-master', 'custom', source='local', path='yolov5-master/face3.pt')
    fr = FaceRecognition()
    fr.encode_faces()
    fr.run_recognition()
