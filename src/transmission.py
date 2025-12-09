import cv2, math
import numpy as np
import matplotlib.pyplot as plt

from Utils import get_size_frame


class OriginVideo:
    def __init__(self , path_video):
        self.path = path_video
        self.B = {
            'origin' : None ,
            'predict': None ,
            'delta'  : None ,
            'receiver': None
        }
        self.G = {
            'origin': None,
            'predict': None,
            'delta': None,
            'receiver': None
        }
        self.R = {
            'origin': None,
            'predict': None,
            'delta': None,
            'receiver': None
        }

        self.delta = None
        self.cnt_uni_vals = dict()

        self.frame_size = None
        self.frame_index = 0
        self.fps_target = 10

        self.height = None
        self.width = None

        self.last_time = cv2.getTickCount()
        self.fps_now = 0

    # ================= SIMPLE PREDICT FUNCTION ================= #
    def predict_next_frame(self, prev_frame, curr_frame):
        prev = prev_frame.astype(np.float32)
        curr = curr_frame.astype(np.float32)

        pred = curr + (curr - prev)
        pred = np.clip(pred, 0, 255).astype(np.uint8)

        return pred
    # ============================================================ #

    def run_exchange_same_channel(self):

        cap = cv2.VideoCapture(self.path)

        ret, frame = cap.read()
        if not ret:
            print("[Error] Can not get the first frame !!!")
            return

        self.height, self.width, _ = frame.shape

        # store original first frame
        self.storage_origin(frame=frame, key='origin')
        self.storage_origin(frame=frame, key='receiver')

        prev_frame = frame.copy()

        while True:

            # read next frame
            ret, frame = cap.read()
            if not ret:
                break

            # ================= APPLY SIMPLE PREDICTION =================
            predicted = self.predict_next_frame(prev_frame, frame)
            # ============================================================

            # ======= YOUR ORIGINAL CHANNEL PROCESSING LOGIC =======
            b = frame[:, :, 0]
            g = frame[:, :, 1]
            r = frame[:, :, 2]

            idx = self.frame_index % 3
            if idx == 0:
                self.delta = b - self.B['receiver']
            elif idx == 1:
                self.delta = g - self.G['receiver']
            else:
                self.delta = r - self.R['receiver']

            self.B['receiver'] += (self.delta * 1.01).astype(np.uint8)
            self.G['receiver'] += (self.delta * 1.01).astype(np.uint8)
            self.R['receiver'] += (self.delta * 1.01).astype(np.uint8)

            out_frame = cv2.merge((self.B['receiver'], self.G['receiver'], self.R['receiver']))
            out_frame = np.clip(out_frame, 0, 255).astype(np.uint8)

            # =======================================================

            # Show ORIGINAL + PREDICTED
            self.show_frame_with_index(out_frame, name="Processed Frame")
            self.show_frame_with_index(predicted, name="Predicted (n+1)")

            prev_frame = frame.copy()
            self.frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------

    def run_simple_same_channel(self):
        cap = cv2.VideoCapture(self.path)
        ret, frame = cap.read()

        if not ret:
            print("[Error] Can not get the first frame !!!")
            return

        self.height, self.width, _ = frame.shape
        self.storage_origin(frame=frame, key='origin')

        while True:
            ret , frame = cap.read()
            if not ret:
                break

            self.storage_origin(frame=frame, key='predict')
            self.B['delta'] = self.calculate_delta('B')
            self.G['delta'] = self.calculate_delta('G') + self.B['delta']
            self.R['delta'] = self.calculate_delta('R') + self.B['delta']

            self.count_unique_values(self.B['delta'])
            self.count_unique_values(self.G['delta'])
            self.count_unique_values(self.R['delta'])

            self.frame_index += 1

        self.plot_pdf("PDF of frequency delta after apply SAME CHANNEL VALUE")

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------

    def run_simple_delta_mask(self):
        cap = cv2.VideoCapture(self.path)
        ret, frame = cap.read()

        if not ret:
            print("[Error] Can not get the first frame !!!")
            return

        self.height, self.width, _ = frame.shape
        self.storage_origin(frame=frame, key='origin')

        while True:
            ret , frame = cap.read()
            if not ret:
                break

            self.storage_origin(frame=frame , key='predict')

            self.B['delta'] = self.calculate_delta('B')
            self.G['delta'] = self.calculate_delta('G')
            self.R['delta'] = self.calculate_delta('R')

            self.count_unique_values(self.B['delta'])
            self.count_unique_values(self.G['delta'])
            self.count_unique_values(self.R['delta'])

            self.frame_index += 1

        self.plot_pdf("PDF of frequency delta before apply SAME CHANNEL VALUE")

        cap.release()
        cv2.destroyAllWindows()

    # ------------------------------------------------------------------------------

    def plot_pdf(self , name , storage_dict=None):
        if storage_dict is None:
            storage_dict = self.cnt_uni_vals

        x = [int(k) for k in storage_dict.keys()]
        y = [int(v) for v in storage_dict.values()]

        plt.bar(x, y)
        plt.xlabel("Value")
        plt.ylabel("Count by log 10")
        plt.title(name)
        plt.show()

    # ------------------------------------------------------------------------------

    def calculate_delta(self , channel):
        if channel == 'R':
            return self.R['predict'] - self.R['origin']
        elif channel == 'B':
            return self.B['predict'] - self.B['origin']
        else:
            return self.G['predict'] - self.G['origin']

    # ------------------------------------------------------------------------------

    def count_unique_values(self , frame ):
        values, counts = np.unique(frame.reshape(-1), return_counts=True)

        for v, c in zip(values, counts):
            self.cnt_uni_vals[v] = self.cnt_uni_vals.get(v , 0) + c

    # ------------------------------------------------------------------------------

    def storage_origin(self , frame , key):
        self.B[key] = frame[:, :, 0]
        self.G[key] = frame[:, :, 1]
        self.R[key] = frame[:, :, 2]

    # ------------------------------------------------------------------------------

    def show_frame_with_index(self , frame , name="Video"):

        now = cv2.getTickCount()
        delta = (now - self.last_time) / cv2.getTickFrequency()
        self.last_time = now

        if delta > 0:
            self.fps_now = 1.0 / delta

        cv2.putText(
            frame,
            f"Frame: {self.frame_index}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"FPS: {self.fps_now:.2f}",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            2
        )

        cv2.imshow(name, frame)
        cv2.waitKey(1)

# ===================================================================================

if __name__ == "__main__":
    origin = OriginVideo("../data/video1.mp4")
    origin.run_exchange_same_channel()
