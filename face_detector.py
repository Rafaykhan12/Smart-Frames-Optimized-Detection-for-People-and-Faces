import face_recognition
import cv2

class face_detector():
    def __init__(self) -> None:
        pass
    
    
    def face_detect(self,frame):
        # Find all the faces and face encodings in the current frame of video
         # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Find all the faces in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)

            
            return face_locations

            # Hit 'q' on the keyboard to quit!
            