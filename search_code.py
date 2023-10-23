import cv2
import numpy as np

camera = cv2.VideoCapture(0)

while (True):
    ret, img = camera.read()
    h, w, c = img.shape

    # 画像を読み込んでグレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)

    # 二値化
    threshold = 180
    ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    cv2.imshow('th', gray)

    # Cannyエッジ検出
    edges = cv2.Canny(gray, 50, 150)
    cv2.imshow('edges', edges)

    # モルフォロジー変換: 膨張
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    cv2.imshow('dilated', dilated)

    # 長い垂直の線を検出するためのモルフォロジー変換
    n = 2
    kernel_long = np.ones((10 * n, 1 * n), np.uint8)
    highlighted = cv2.erode(dilated, kernel_long, iterations=1)
    cv2.imshow('highlighted', highlighted)

    # モルフォロジー変換: 膨張
    kernel = np.ones((3, 3), np.uint8)
    x = cv2.dilate(highlighted, kernel, iterations=3)
    cv2.imshow('x', x)

    # 輪郭を見つける
    contours, _ = cv2.findContours(
        # dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # highlighted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # バーコードのような領域を描画
    min_area = (h / 10) * (w / 10)
    max_area = (h / 2) * (w / 2)
    for contour in contours:
        if (min_area <= cv2.contourArea(
                contour)) and (cv2.contourArea(
                contour) < max_area):  # この値は画像のサイズや内容に応じて調整する必要があるかもしれません
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if 2 * h <= w:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # 結果を表示
    cv2.imshow('Barcode Detection', img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
