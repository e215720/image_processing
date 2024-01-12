import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.kalman = cv2.KalmanFilter(state_dim, measurement_dim, 0)
        self.kalman.transitionMatrix = np.eye(state_dim, dtype=np.float32)
        self.kalman.measurementMatrix = np.eye(measurement_dim, state_dim, dtype=np.float32)
        self.kalman.processNoiseCov = 1e-5 * np.eye(state_dim, dtype=np.float32)
        self.kalman.measurementNoiseCov = 1e-1 * np.eye(measurement_dim, dtype=np.float32)
        self.kalman.errorCovPost = np.eye(state_dim, dtype=np.float32)

    def update(self, measurement):
        prediction = self.kalman.predict()
        estimate = self.kalman.correct(measurement)
        return estimate

# 画像から輪郭を検出する関数
def contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_binary = cv2.threshold(img_gray, 60, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])
            return x, y
    return None

movie = cv2.VideoCapture('img/testmovie1.avi')

# 動画ファイル保存用の設定
fps = int(movie.get(cv2.CAP_PROP_FPS))
w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('img/video_out1.mp4', fourcc, fps, (w, h), True)

# カルマンフィルタの初期化
state_dim = 4  # x, y 座標と速度の4次元
measurement_dim = 2  # x, y 座標の2次元
kf = KalmanFilter(state_dim, measurement_dim)

# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
while True:
    ret, frame = movie.read()

    if not ret:
        break

    center = contours(frame)

    if center is not None:
        x, y = center

        # カルマンフィルタを使用して予測と修正を行う
        measurement = np.array([x, y], dtype=np.float32)
        state = kf.update(measurement)

        # トラッキング結果を描画
        x, y = state[:2]  # 最初の2つの要素を取得
        frame = cv2.circle(frame, (int(x), int(y)), 30, (0, 255, 0), 3)

        video.write(frame)  # 動画を保存する

# 動画オブジェクト解放
movie.release()
video.release()
cv2.destroyAllWindows()
