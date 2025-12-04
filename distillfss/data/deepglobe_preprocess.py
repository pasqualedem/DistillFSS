import os
import click
import cv2
import numpy as np
from PIL import Image

def cut_images(input_folder, output_folder, num=6):
    image_folder = os.path.join(output_folder, "images")
    label_folder = os.path.join(output_folder, "labels")
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)
    
    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        img = cv2.imread(path)
        hei, wid = img.shape[:2]

        for i in range(num):
            for j in range(num):
                hei_0, hei_1 = int(i * hei / num), int((i + 1) * hei / num)
                wid_0, wid_1 = int(j * wid / num), int((j + 1) * wid / num)
                roiImg = img[hei_0:hei_1, wid_0:wid_1]
                
                filename, ext = os.path.splitext(f)
                output_path = image_folder if ext == '.jpg' else label_folder
                output_file = os.path.join(output_path, f"{filename}_{i}{j}{ext}")
                cv2.imwrite(output_file, roiImg)
    print('1st stage end.')

def filter_masks(input_folder, output_folder, min_percent=0.048, total_pixels=166464):
    filtered_folder = os.path.join(output_folder, "filtered_masks")
    os.makedirs(filtered_folder, exist_ok=True)
    
    for f in os.listdir(input_folder):
        path = os.path.join(input_folder, f)
        img = Image.open(path)
        colors = img.getcolors()

        if len(colors) > 1 and min(num[0] / total_pixels for num in colors) > min_percent:
            img.save(os.path.join(filtered_folder, f))
    print('2nd stage end.')

def generate_binary_masks(input_folder, output_folder):
    labelset = [[0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0], [0, 0, 255], [255, 255, 255]]
    masklist = os.listdir(input_folder)
    binary_folder = os.path.join(output_folder, "binary_masks")
    os.makedirs(binary_folder, exist_ok=True)
    
    def get_binary_map(img, label):
        mask = np.zeros_like(img)
        isnull = True
        for j in range(img.shape[0]):
            for k in range(img.shape[1]):
                if (img[j, k] == labelset[label]).all():
                    mask[j, k] = [255, 255, 255]
                    isnull = False
        return isnull, mask

    for label in range(6):
        label_folder = os.path.join(binary_folder, str(label + 1))
        os.makedirs(label_folder, exist_ok=True)
        filenames = []
        
        for mask_file in masklist:
            if mask_file.endswith('.png'):
                img = cv2.imread(os.path.join(input_folder, mask_file))
                isnull, binary_mask = get_binary_map(img, label)
                if not isnull:
                    filenames.append(os.path.splitext(mask_file)[0])
                    cv2.imwrite(os.path.join(label_folder, mask_file), binary_mask)
        
        with open(os.path.join(binary_folder, f"{label+1}.txt"), 'w') as file:
            file.writelines(f"{name}\n" for name in filenames)
    print('3rd stage end.')

@click.command()
@click.option('--input-folder', required=True, type=click.Path(exists=True), help='Path to input data folder')
@click.option('--output-folder', required=True, type=click.Path(), help='Path to save processed data')
@click.option('--num', default=6, type=int, help='Number of subdivisions')
@click.option('--min-percent', default=0.048, type=float, help='Minimum foreground percentage')
def main(input_folder, output_folder, num, min_percent):
    cut_images(input_folder, output_folder, num)
    filter_masks(os.path.join(output_folder, "labels"), output_folder, min_percent)
    generate_binary_masks(os.path.join(output_folder, "filtered_masks"), output_folder)

if __name__ == '__main__':
    main()
