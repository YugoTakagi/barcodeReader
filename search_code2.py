import cv2
import numpy as np

from mybarcode import MyBarcodeReader2


def get_code(img):
    h, w, c = img.shape
    # 画像を読み込んでグレースケール変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    # 二値化
    threshold = 200
    ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    # cv2.imshow('th', gray)

    # モルフォロジー変換: 膨張
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(gray, kernel, iterations=3)
    # cv2.imshow('erode', dilated)

    # 輪郭を見つける
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # バーコードのような領域を描画
    min_area = (h / 10) * (w / 10)
    max_area = (h / 2) * (w / 2)
    imgs_ = []
    for contour in contours:
        if (min_area <= cv2.contourArea(
                contour)) and (cv2.contourArea(
                contour) < max_area):  # この値は画像のサイズや内容に応じて調整する必要があるかもしれません
            x, y, w, h = cv2.boundingRect(contour)
            # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # if (2 * h <= w):
            if (h <= w):
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                imgs_.append(img[y:(y + h), x:(x + w)])
    # for i in range(len(imgs_)):
    #     cv2.imshow(f'{i}', imgs_[i])

    return imgs_, img


def main():
    camera = cv2.VideoCapture(0)
    mbr = MyBarcodeReader2()

    while (True):
        ret, img = camera.read()
        frame = cv2.flip(src=img, flipCode=1)

        imgs, z = get_code(img)
        for i in range(len(imgs)):
            h, w, c = imgs[i].shape
            n = 5
            x = cv2.resize(imgs[i], (int(w * n * 0.8), int(h * n * 2)))
            y = mbr.xyz(x)
            pt1 = (100, 100)
            cv2.putText(z,
                        mbr.text,
                        pt1,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        mbr.color_palet[mbr.i],
                        2)

            # cv2.imshow("img", img)
            print(f'text is {mbr.text}')

            cv2.imshow("y", y)

        cv2.imshow("frame", z)
        # 結果を表示
        # for i in range(len(imgs)):
        #     cv2.imshow(f'{i}', imgs[i])
        # cv2.imshow('Barcode Detection', img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
