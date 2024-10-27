import face_recognition
import cv2
import numpy as np
import os
class face_recognizer():
    def __init__(self) -> None:
   

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
         # Initialize lists to hold face encodings and corresponding names
        self.known_face_encodings = []
        self.known_face_names = []
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


    def face_recog(self,frame,face_locations):
        small_frame=frame
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]

            face_names.append(name)
        return face_names