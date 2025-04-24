import cv2
import os
from pathlib import Path

def sobel(src, ksize=-1, scale=1, delta=0, ddepth=cv2.CV_16S):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(src_gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(src_gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def process_folder_sobel(folder_path, ksize, scale, delta, ddepth):
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

            grad = sobel(image, ksize, scale, delta, ddepth)
            save_path = str(image_path)  # overwriting original
            cv2.imwrite(save_path, grad)
            print(f"Processed and saved: {save_path}")

def main():
    folders = ["test2017", "val2017", "train2017"]
    ksize = -1
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S

    for folder in folders:
        process_folder_sobel(folder, ksize, scale, delta, ddepth)

if __name__ == "__main__":
    main()