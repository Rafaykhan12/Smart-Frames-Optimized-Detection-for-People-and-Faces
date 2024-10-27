from dlib_face_recognsition_2people_frames_skipping import recognition

def main():

    channel_no=1
    facial_recognition=True
    frame_skip=3
    video_source=["train_images/rafay_video.mp4"]
    model= recognition(channel_no,video_source,facial_recognition,frame_skip)

    model.start_algo()

if __name__ == "__main__":
    main()