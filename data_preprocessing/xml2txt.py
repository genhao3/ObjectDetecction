import os
import xml.etree.ElementTree as ET

from tqdm import tqdm

def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


classes = ['person','bird', 'cat', 'cow', 'dog', 'horse', 'sheep','aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']


DATASET_PATH = 'F:\\360Downloads\VOCtrainval_11-May-2012\VOCdevkit\VOC2012'



filenames = os.listdir(os.path.join(DATASET_PATH, 'Annotations'))

save_dir = os.path.join(DATASET_PATH, 'Detectionlabels')

os.makedirs(save_dir,exist_ok=False) #False 不会覆盖

for filename in tqdm(filenames):
    filename_path = os.path.join(DATASET_PATH,'Annotations',filename)
    f = open(filename_path)
    out_file = open(os.path.join(save_dir,filename.replace('xml','txt')),'w')

    tree = ET.parse(f)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bndbox = convert((w, h), points)
        out_file.write(str(cls_id) + ' ' + ' '.join([str(a) for a in bndbox]) + '\n')