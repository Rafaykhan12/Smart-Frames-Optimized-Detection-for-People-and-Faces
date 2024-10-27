import cv2
import numpy as np
import queue
import threading

from cv_models import cv_models


class recognition():
    def __init__(self,channel,channel_video,facial_recognition,frame_skip):
        # Frame skipping parameters
        self.frame_skip = frame_skip
        self.frame_queue = queue.Queue(maxsize=0)
        self.channel=channel
        self.channel_video=channel_video
        self.facial_recognition=facial_recognition
    def capture_frames(self,video_source):
        video_capture = cv2.VideoCapture(video_source)
        frame_count = 0

        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                if frame_count % self.frame_skip == 0:
                    self.frame_queue.put(frame)
                frame_count += 1
            else:
                break

        video_capture.release()

    def process_frames(self):
        model=cv_models()
        # face_recog_algo=face_recognizer()
        
        
        target_width = 640  # Set the desired width
        target_height = 480  # Set the desired height
        while True:
            if not self.frame_queue.empty():
                faces_names=['unknown']
                frame = self.frame_queue.get()                       
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                # Resize frame of video to 1/4 size for faster face detection processing
                frame,person_detected_flag=model.count_people(frame)
                if person_detected_flag==1:
                    face_locations=model.face_detect(frame)
                    if not face_locations:
                        print("face not detected")
                    else:    
                        if self.facial_recognition==True :
                            faces_names=model.face_recog(small_frame,face_locations)
                        # Display the results
                        for (top, right, bottom, left), name in zip(face_locations, faces_names):
                                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                            top *= 4
                            right *= 4
                            bottom *= 4
                            left *= 4

                            # Draw a box around the face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                            # Draw a label with a name below the face
                            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                            font = cv2.FONT_HERSHEY_DUPLEX
                            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

                else:
                    print(" No person/faces are detected")    
                # Display the resulting image
                resized_frame = cv2.resize(frame, (target_width, target_height))
                cv2.imshow(str(self.channel), resized_frame)
                print("channel no:",self.channel)
               
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break    


    def start_algo(self):
       
        capture_thread = threading.Thread(target=self.capture_frames, args=(self.channel_video[0],))
        display_thread = threading.Thread(target=self.process_frames, args=())

  # Start threads
        capture_thread.start()
        display_thread.start()

        # Wait for threads to finish
        capture_thread.join()
        display_thread.join()

        # Close OpenCV window
        cv2.destroyAllWindows()




