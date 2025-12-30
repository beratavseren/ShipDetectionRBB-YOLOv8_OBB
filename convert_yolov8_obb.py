
import os
import os.path as osp
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as np
import shutil

#colab de yaptığım için yollar bu şekilde

SOURCE_XML_DIR = r"/content/data/HRSC2016/HRSC2016/FullDataSet/Annotations"
SOURCE_IMG_DIR = r"/content/data/HRSC2016/HRSC2016/FullDataSet/AllImages"
OUTPUT_DIR = r"/content/yolov8_obb/ship_dataset_yolo"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def convert_xml_to_yolo_obb():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for s in ['train', 'val']:
        ensure_dir(osp.join(OUTPUT_DIR, 'images', s))
        ensure_dir(osp.join(OUTPUT_DIR, 'labels', s))

    xml_files = sorted([f for f in os.listdir(SOURCE_XML_DIR) if f.endswith('.xml')])
    split = int(len(xml_files) * 0.8)
    splits = {'train': xml_files[:split], 'val': xml_files[split:]}

    for subset, files in splits.items():
        print(f"{subset.upper()} hazırlanıyor...")
        for xml_file in files:
            file_id = xml_file.replace('.xml', '')
            img_path = None
            img_ext = None
            for ext in ['.bmp', '.jpg', '.png', '.tif']:
                p = osp.join(SOURCE_IMG_DIR, file_id + ext)
                if os.path.exists(p):
                    img_path = p
                    img_ext = ext
                    break
            if img_path is None:
                continue

            img = cv2.imread(img_path)
            h_img, w_img = img.shape[:2]

            tree = ET.parse(osp.join(SOURCE_XML_DIR, xml_file))
            root = tree.getroot()
            objs = root.findall('.//HRSC_Object')

            lines = []
            for obj in objs:
                try:
                    cx = float(obj.find('mbox_cx').text)
                    cy = float(obj.find('mbox_cy').text)
                    w = float(obj.find('mbox_w').text)
                    h = float(obj.find('mbox_h').text)
                    ang = float(obj.find('mbox_ang').text)
                except:
                    continue

                angle_deg = ang * 180.0 / math.pi
                rect = ((cx, cy), (w, h), angle_deg)
                pts = cv2.boxPoints(rect) # 4 nokta
                
                normalized_pts = []
                for p in pts:
                    normalized_pts.append(p[0] / w_img)
                    normalized_pts.append(p[1] / h_img)
                
                line = "0 " + " ".join([f"{p:.6f}" for p in normalized_pts]) + "\n"
                lines.append(line)

            if len(lines) == 0:
                continue

            cv2.imwrite(osp.join(OUTPUT_DIR, 'images', subset, file_id + '.jpg'), img)

            with open(osp.join(OUTPUT_DIR, 'labels', subset, file_id + '.txt'), 'w') as f:
                f.writelines(lines)

        print(f"{subset} tamamlandı.")

    print(f"Veriler '{OUTPUT_DIR}' klasöründe.")

if __name__ == "__main__":
    convert_xml_to_yolo_obb()
