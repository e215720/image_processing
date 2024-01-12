import cv2

# 動画読み込みの設定
movie = cv2.VideoCapture('img/testmovie1.avi')

# 動画ファイル保存用の設定
fps = int(movie.get(cv2.CAP_PROP_FPS))
w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter('img/video_out.mp4', fourcc, fps, (w, h), isColor=False)  # isColorをFalseに設定

# 背景差分の設定
fgbg = cv2.createBackgroundSubtractorMOG2()  # createBackgroundSubtractorMOG2()を使用

# ファイルからフレームを1枚ずつ取得して動画処理後に保存する
while True:
    ret, frame = movie.read()
    if not ret:
        break

    fgmask = fgbg.apply(frame)
    video.write(fgmask)

# 撮影用オブジェクトとウィンドウの解放
movie.release()
video.release()
cv2.destroyAllWindows()
