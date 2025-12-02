from PIL import Image
from yolo import YOLO
import os
from tqdm import tqdm

if __name__ == "__main__":
    #----------------------------------------------------------------------------------------------------------#
    #    mode is used to specify the mode of the test:
    #   'predict' means single picture prediction
    #   'dir_predict' means traverse the folder to test and save.
    #----------------------------------------------------------------------------------------------------------#
    mode = "dir_predict"
    crop            = False
    count           = False
    
    dir_origin_path = ""
    dir_save_path   = ""
    yolo = YOLO()
    if mode == "predict":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "dir_predict":
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
