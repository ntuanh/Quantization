import cv2
import numpy as np

from Utils import get_size_frame

class OriginVideo:
    def __init__(self , path_video):
        self.path = path_video
        self.R = []
        self.G = []
        self.B = []
        self.R_predict = []
        self.G_predict = []
        self.B_predict = []
        self.delta = None
        self.B_previous = None
        self.G_previous = None
        self.R_previous = None

        self.frame_size = None
        self.frame_index = 0
        self.fps = 40

        self.height =  None
        self.width = None

    def run(self):
        cap = cv2.VideoCapture(self.path)
        # fps_input = cap.get(cv2.CAP_PROP_FPS)
        # print(f"[FPS INPUT] {fps_input}")
        ret, frame = cap.read()
        if ret:
            self.height, self.width, _ = frame.shape
            self.B_previous = frame[:, :, 0]
            self.G_previous = frame[:, :, 1]
            self.R_previous = frame[:, :, 2]
        else :
            print("[Error] Can not get the first frame !!!")

        while True :
            # Read frame input
            ret , frame = cap.read()
            if not ret:
                break

            # get delta
            self.B = frame[:, :, 0]     # 1 for G and 2 for R
            self.G = frame[:, :, 1]
            self.R = frame[:, :, 2]
            self.delta = self.B - self.B_previous
            self.B_previous = self.B

            # predict
            self.G_predict = np.abs(self.delta + self.G_previous)
            self.G_previous = self.G_predict
            self.R_predict = np.abs(self.delta + self.R_previous)
            self.R_previous = self.R_predict

            # merge
            # predict_frame = np.stack((self.B , self.G_predict ,self.R_predict))
            # predict_frame = np.transpose(predict_frame , (1 , 2, 0))
            predict_frame = cv2.merge((self.B , self.G_predict ,self.R_predict))
            delta_frame = predict_frame-frame
            # print(predict_frame.shape)
            delta_G = np.abs(self.G - self.G_predict)
            delta_R = np.abs(self.R - self.R_predict)

            # show
            self.frame_index += 1
            self.show_frame_with_index(delta_G , "G")
            self.show_frame_with_index(delta_R , "R")
            # self.show_frame_with_index(frame)
            # self.show_frame_with_index(frame , "Video 2")

        cap.release()
        cv2.destroyAllWindows()

    def show_frame_with_index(self , frame , name="Video"):
        cv2.putText(
            frame,
            f"Frame: {self.frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow(name, frame)
        cv2.waitKey(int(1000 / self.fps))


if __name__ == "__main__":
    origin = OriginVideo("../data/video1.mp4")
    origin.run()