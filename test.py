#https://imageai.readthedocs.io/en/latest/video/

from imageai.Detection import VideoObjectDetection
import os

execution_path = os.getcwd()

detector = VideoObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("yolov3.pt")
detector.loadModel()

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")

video_path = detector.detectObjectsFromVideo(input_file_path="video1.mp4", output_file_path="testdetection3", 
            frames_per_second=30, per_frame_function=forFrame, log_progress=True)
print(video_path)