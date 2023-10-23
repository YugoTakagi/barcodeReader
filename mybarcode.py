import numpy as np
from pyzbar.pyzbar import decode
import cv2
import pyperclip


def test(img):
    # コントラストと明るさを調整
    adjusted = cv2.convertScaleAbs(img, alpha=1.5, beta=0.)

    # 二値化を行い、黒と白をはっきりと分ける
    _, binary = cv2.threshold(adjusted, 128, 255, cv2.THRESH_BINARY)
    cv2.imshow('a', adjusted)
    cv2.imshow('b', binary)


class MyBarcodeReader2:
    def __init__(self) -> None:
        self._text = ""
        self.text = ""
        # self.color_palet = [(41, 15, 22), (115, 106, 36),
        # (139, 143, 54), (193, 223, 243), (168, 190, 221)]
        self.color_palet = [(82, 99, 238), (227, 178, 8),
                            (244, 233, 239), (115, 167, 87), (109, 77, 72)]
        self.i = 0

    def xyz(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hist, bins = np.histogram(img_gray.flatten(), 256, [0,256])

        # ヒストグラム平坦化
        # equ = cv2.equalizeHist(img_gray)
        # res = np.hstack((img, equ))

        # ガウシアンフィルタ
        # equ = cv2.GaussianBlur(equ, (5, 5), 0)
        # equ = cv2.medianBlur(equ, 5)

        # 先鋭化
        # k = 1
        # kernel = np.array([[-k / 9, -k / 9, -k / 9],
        #                    [-k / 9, 1 + 8 * k / 9, k / 9],
        #                    [-k / 9, -k / 9, -k / 9]], np.float32)
        # equ = cv2.filter2D(equ, -1, kernel)

        # 二値化
        # threshold = int(255 / 2)
        # ret, equ = cv2.threshold(equ, threshold, 255, cv2.THRESH_BINARY)

        # _h, _w = equ.shape
        # equ = cv2.resize(equ, (int(_w * 0.3), int(_h)))

        # コントラストと明るさを調整
        adjusted_img = cv2.convertScaleAbs(img_gray, alpha=1.5, beta=0.)

        # 3x3のカーネルを定義
        kernel = np.ones((3, 3), np.uint8)
        # オープニング
        adjusted_img = cv2.morphologyEx(
            adjusted_img, cv2.MORPH_OPEN, kernel, iterations=2)

        # 収縮
        adjusted_img = cv2.erode(adjusted_img, kernel=kernel, iterations=2)

        # メディアンフィルタ
        # adjusted_img = cv2.medianBlur(adjusted_img, 3)

        # 二値化を行い、黒と白をはっきりと分ける
        # _, adjusted_img = cv2.threshold(
        #     adjusted_img, 180, 255, cv2.THRESH_BINARY)

        h, w, c = img.shape
        # Barcode
        barcodes = decode(adjusted_img)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            # バーコードやQRコードの周りに矩形を描く
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # バーコードのデータとバーコードのタイプを画像上に書く
            barcode_data = barcode.data.decode("utf-8")
            # barcode_type = barcode.type
            # text = f"{barcode_data} ({barcode_type})"
            text = f"{barcode_data}"
            # cv2.putText(equ, text, (x, y - 10),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self._text = self.text
            self.text = text
            if self._text != self.text:
                self.i += 1
                self.i = self.i % len(self.color_palet)
                pyperclip.copy(self.text)

        return adjusted_img


class MyBarcodeReader:
    def __init__(self) -> None:
        self._text = ""
        self.text = ""
        # self.color_palet = [(41, 15, 22), (115, 106, 36),
        # (139, 143, 54), (193, 223, 243), (168, 190, 221)]
        self.color_palet = [(82, 99, 238), (227, 178, 8),
                            (244, 233, 239), (115, 167, 87), (109, 77, 72)]
        self.i = 0

    def xyz(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # hist, bins = np.histogram(img_gray.flatten(), 256, [0,256])

        # ヒストグラム平坦化
        equ = cv2.equalizeHist(img_gray)
        # res = np.hstack((img, equ))

        # ガウシアンフィルタ
        equ = cv2.GaussianBlur(equ, (5, 5), 0)
        # equ = cv2.medianBlur(equ, 5)

        # 先鋭化
        k = 1
        kernel = np.array([[-k / 9, -k / 9, -k / 9],
                           [-k / 9, 1 + 8 * k / 9, k / 9],
                           [-k / 9, -k / 9, -k / 9]], np.float32)
        equ = cv2.filter2D(equ, -1, kernel)

        # 二値化
        threshold = int(255 / 2)
        ret, equ = cv2.threshold(equ, threshold, 255, cv2.THRESH_BINARY)

        _h, _w = equ.shape
        equ = cv2.resize(equ, (int(_w * 0.3), int(_h)))

        h, w, c = img.shape
        # Barcode
        barcodes = decode(equ)
        for barcode in barcodes:
            (x, y, w, h) = barcode.rect
            # バーコードやQRコードの周りに矩形を描く
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # バーコードのデータとバーコードのタイプを画像上に書く
            barcode_data = barcode.data.decode("utf-8")
            # barcode_type = barcode.type
            # text = f"{barcode_data} ({barcode_type})"
            text = f"{barcode_data}"
            cv2.putText(equ, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            self._text = self.text
            self.text = text
            if self._text != self.text:
                self.i += 1
                self.i = self.i % len(self.color_palet)
                pyperclip.copy(self.text)

        return equ


if __name__ == "__main__":
    mbr = MyBarcodeReader2()
    camera = cv2.VideoCapture(0)

    while (True):
        ret, frame = camera.read()

        # test(frame)

        h, w, c = frame.shape
        wc = w / 2
        hc = h / 2
        pt1 = (int(wc - wc / 3), int(hc - hc / 6))
        pt2 = (int(wc + wc / 3), int(hc + hc / 6))
        cv2.rectangle(frame, pt1, pt2, mbr.color_palet[mbr.i], 5)

        _frame = frame[pt1[1]:pt2[1], pt1[0]:pt2[0]]
        _frame = cv2.resize(_frame, (int(w * 0.8), int(h * 0.8)))

        img = mbr.xyz(_frame)

        frame = cv2.flip(src=frame, flipCode=1)
        cv2.putText(frame, mbr.text, pt1,
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, mbr.color_palet[mbr.i], 2)

        cv2.imshow("frame", frame)
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    # barcodes = pyzbar.pyzbar.decode(frame)
