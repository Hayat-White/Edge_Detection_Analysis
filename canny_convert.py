import cv2
import os
from pathlib import Path

def sobel(src, ksize, scale, delta, ddepth):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(src_gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src_gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def canny(src, low_threshold, ratio=2):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.blur(src_gray, (3, 3))
    detected_edges = cv2.Canny(blurred, low_threshold, low_threshold * ratio, apertureSize=3)
    dst = cv2.bitwise_and(src, src, mask=detected_edges)
    return dst

def process_folder(folder_path, low_threshold):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder_path}")
        return

    for image_path in folder.glob("*.*"):
        if image_path.is_file():
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            result = canny(image, low_threshold)
            cv2.imwrite(str(image_path), result)
            print(f"Processed and saved: {image_path}")

def main():
    folders = ["test2017", "val2017", "train2017"]
    low_threshold = 10

    for folder in folders:
        process_folder(folder, low_threshold)

if __name__ == "__main__":
    main()