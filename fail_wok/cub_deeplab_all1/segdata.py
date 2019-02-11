import os
import os.path
import cv2
import numpy as np

from torch.utils.data import Dataset
cv2.ocl.setUseOpenCL(False)


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(split='train', data_root=None, data_list=None):
    assert split in ['train', 'val', 'test']
    if not os.path.isfile(data_list):
        raise (RuntimeError("Image list file do not exist: " + data_list + "\n"))
    image_label_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    print("Starting Checking image&label pair {} list...".format(split))
    for line in list_read:
        line = line.strip()
        line_split = line.split(' ')
        if split == 'test':
            if len(line_split) != 1:
                raise (RuntimeError("Image list file read line error : " + line + "\n"))
            image_name = os.path.join(data_root, line_split[0])
            label_name = image_name  # just set place holder for label_name, not for use
        else:
            image_name = os.path.join(data_root, line_split[0])
            label_name = os.path.join(data_root, line_split[1])
            att = [float(item) for item in line_split[2:]]
        '''
        following check costs some time
        if is_image_file(image_name) and is_image_file(label_name) and os.path.isfile(image_name) and os.path.isfile(label_name):
            item = (image_name, label_name)
            image_label_list.append(item)
        else:
            raise (RuntimeError("Image list file line error : " + line + "\n"))
        '''
        item = (image_name, label_name, att)
        image_label_list.append(item)
    print("Checking image&label pair {} list done!".format(split))
    return image_label_list


class SegData(Dataset):
    def __init__(self, split='train', data_root=None, data_list=None, transform=None):
        self.split = split
        self.data_list = make_dataset(split, data_root, data_list)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        image_path, label_path, att = self.data_list[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # BGR 3 channel ndarray wiht shape H * W * 3
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # convert cv2 read image from BGR order to RGB order
        image = np.float32(image)
        label = (cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)>150).astype('uint8')  # GRAY 1 channel ndarray with shape H * W
        if image.shape[0] != label.shape[0] or image.shape[1] != label.shape[1]:
            raise (RuntimeError("Image & label shape mismatch: " + image_path + " " + label_path + "\n"))
        if self.transform is not None:
            image, label = self.transform(image, label)
        bill = att[0:9]+att[149:152]+att[278:293] #27
        wing = att[9:24]+att[212:217]+att[308:312] #24
        upperparts = att[24:54] #30
        breast = att[54:58] + att[105:120] #19
        back = att[58:73] + att[236:240] #19
        tail = att[73:94]+att[167:182]+att[240:244] #40
        head = att[94:105] #11
        throat = att[120:135] #15
        eye = att[135:149]+att[212:217]+att[308:312] #23
        forehead = att[152:167] #15
        nape = att[182:197] #15
        belly = att[197:212]+att[244:248] #19
        leg = att[263:278] #15
        crown = att[293:308] #15
        #print(np.array(bill +wing + upperparts+ breast+ back+ tail+ head+ throat+ eye+ forehead+ nape+ belly+ leg+ crown))
        return image, label, np.array(bill +wing + upperparts+ breast+ back+ tail+ head+ throat+ eye+ forehead+ nape+ belly+ leg+ crown)
