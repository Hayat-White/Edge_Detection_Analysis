import cv2
import numpy as np
from pathlib import Path

def prewitt(src):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # Define Prewitt kernels
    kernelx = np.array([[ -1, 0, 1],
                        [ -1, 0, 1],
                        [ -1, 0, 1]], dtype=np.float32)
    
    kernely = np.array([[ 1,  1,  1],
                        [ 0,  0,  0],
                        [-1, -1, -1]], dtype=np.float32)

    # Apply filters
    grad_x = cv2.filter2D(src_gray, cv2.CV_16S, kernelx)
    grad_y = cv2.filter2D(src_gray, cv2.CV_16S, kernely)

    # Convert to absolute values
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # Combine gradients
    grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    return grad

def process_folder_prewitt(folder_path):
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

            edge_output = prewitt(image)
            save_path = str(image_path)  # overwrite original
            cv2.imwrite(save_path, edge_output)
            print(f"Processed and saved: {save_path}")

def main():
    folders = ["test2017", "val2017", "train2017"]

    for folder in folders:
        process_folder_prewitt(folder)

if __name__ == "__main__":
    main()
