import torch
import cv2
import face_recognition
import numpy as np
import os
class cv_models():

    def __init__(self):
        
        # Load the YOLOv5 model (pretrained)
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

        # Set the class index for "person" (YOLOv5 class 0 is for 'person')
        self.person_class_index = 0      

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
         # Initialize lists to hold face encodings and corresponding names
        self.known_face_encodings = []
        self.known_face_names = []
        self.person_detected_flag=0
        self.load_faces()
    def load_faces(self):
                # Path to the directory containing images
        image_directory = "./train_images"

       

        # Loop through each file in the directory
        for filename in os.listdir(image_directory):
            if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
                # Load the image file
                image_path = os.path.join(image_directory, filename)
                image = face_recognition.load_image_file(image_path)
                
                # Get the face encoding for the image
                face_encodings = face_recognition.face_encodings(image)
                if face_encodings:
                    # Assuming each image contains exactly one face
                    self.known_face_encodings.append(face_encodings[0])
                    # Use the filename (without extension) as the name
                    self.known_face_names.append(os.path.splitext(filename)[0])

    def count_people(self,frame):




        # Run YOLOv5 inference on the frame
        results = self.model(frame)

        # Filter results to keep only detections of persons (class index 0)
        person_detections = results.xyxy[0][results.xyxy[0][:, -1] == self.person_class_index]

        # Check if there are any person detections
        if person_detections.size(0) == 0:
            print("Person not detected")
            self.person_detected_flag = 0
        else:    
            # person_roi_list=[]
            for detection in person_detections:
                    
                x1, y1, x2, y2, conf, cls = map(int, detection[:6])
                person_roi = frame[y1:y2, x1:x2]
                label = f'Person {conf:.2f}'
                self.person_detected_flag = 1

                
                
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # # Put label text
                # cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame
        # cv2.imshow('YOLOv5 Person Detection', frame)

        # Write the frame to the output video
        # out.write(frame)
        return frame,self.person_detected_flag
    
    def face_detect(self,frame):
         # Find all the faces and face encodings in the current frame of video
         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

            
        return face_locations
    
    def face_recog(self,frame,face_locations):
        small_frame=frame
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                face_names.append(name)  

        return face_names