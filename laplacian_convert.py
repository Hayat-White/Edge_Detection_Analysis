import cv2
from pathlib import Path

def laplace(src, ksize=3, ddepth=cv2.CV_16S):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    dst = cv2.Laplacian(src_gray, ddepth, ksize=ksize)
    abs_dst = cv2.convertScaleAbs(dst)
    return abs_dst

def process_folder_laplace(folder_path, ksize, ddepth):
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

            edge_output = laplace(image, ksize, ddepth)
            save_path = str(image_path)  # overwrite original image
            cv2.imwrite(save_path, edge_output)
            print(f"Processed and saved: {save_path}")

def main():
    folders = ["test2017", "val2017", "train2017"]
    ksize = 3
    ddepth = cv2.CV_16S

    for folder in folders:
        process_folder_laplace(folder, ksize, ddepth)

if __name__ == "__main__":
    main()