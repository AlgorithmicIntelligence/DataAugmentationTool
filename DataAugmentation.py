import os
import cv2
import numpy as np
import sys

import tkinter as tk
from tkinter import *
from tkinter import filedialog
import threading

def get_img_and_bounding_boxes(img_path, label_path):
    img = cv2.imread(img_path)
    try:
        f = open(label_path, 'r')
    except:
        print("Can't Find",label_path)

    h, w = img.shape[:2]

    bounding_boxes = []
    for line in f:
        label = np.array(line.split(), dtype=np.float32)
        bounding_box = np.full(5, -1, dtype=np.float32)
        bounding_box[0] = label[0]
        bounding_box[1] = (label[1] - label[3]/2) * w
        bounding_box[2] = (label[2] - label[4]/2) * h
        bounding_box[3] = min((label[1] + label[3]/2) * w, w-1)
        bounding_box[4] = min((label[2] + label[4]/2) * h, h-1)
        
        bounding_boxes.append(bounding_box)

    bounding_boxes = np.array(bounding_boxes, dtype=np.float32)

    return img, bounding_boxes

def color_jitter(img):
    img = img/255.
    img_avg = np.average(img, axis=(0, 1))
    img_std = np.std(img, axis=(0, 1))
    img_norm = (img - img_avg) / img_std
    img_cov = np.zeros((3, 3))
    for data in img_norm.reshape(-1, 3):
        img_cov += data.reshape(3, 1) * data.reshape(1, 3)
    img_cov /= len(img_norm.reshape(-1, 3))
    
    eig_values, eig_vectors = np.linalg.eig(img_cov)
    alphas = np.random.normal(0, 0.1, 3)
    img_reconstruct_norm = img_norm + np.sum((eig_values * alphas) * eig_vectors, axis=1)
    img_reconstruct = img_reconstruct_norm * img_std + img_avg
    img_reconstruct *= 255.
    img_reboundary = np.maximum(np.minimum(img_reconstruct , 255), 0).astype(np.uint8)
    return img_reboundary

def hsv_augmentation(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv = img_hsv.astype(np.float32)
    
    s_ratio = np.random.uniform(0.2, 1.8)
    img_hsv[..., 1] *= s_ratio
    img_hsv[img_hsv[..., 1] > 255, 1] = 255
    v_ratio = np.random.uniform(0.2, 1.8)
    img_hsv[..., 2] *= v_ratio
    img_hsv[img_hsv[..., 2] > 255, 2] = 255
    img_hsv = cv2.cvtColor((img_hsv+0.5).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return img_hsv

def rotation(img, bounding_boxes, x_range=[-50, 50], y_range=[-50, 50], z_range=[-60, 60]):
    '''

    x_range : rotate from top to bottom.
    y_range : rotate from left to right.
    z_range : rotate with 2D.

    '''

    theta_x = np.random.uniform(*x_range)
    theta_y = np.random.uniform(*y_range)
    theta_z = np.random.uniform(*z_range)

    theta_x = theta_x * np.pi / 180
    theta_y = theta_y * np.pi / 180
    theta_z = theta_z * np.pi / 180

    matrix_x = np.array([[1, 0, 0], [0, np.cos(theta_x), -np.sin(theta_x)], [0, np.sin(theta_x), np.cos(theta_x)]])
    matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    matrix_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    rotate_matrix = np.dot(np.dot(matrix_x, matrix_y), matrix_z)[:2, :2]

    h, w = img.shape[:2]
    corners_img = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    corners_img_r = np.dot(rotate_matrix, corners_img.T)
    corners_min = np.min(corners_img_r, axis=1)
    corners_max = np.max(corners_img_r, axis=1)

    bounding_boxes_warp = list()
    for bounding_box in bounding_boxes:
        corners = np.array([[bounding_box[1],bounding_box[2]],[bounding_box[3],bounding_box[2]],
                            [bounding_box[3],bounding_box[4]],[bounding_box[1],bounding_box[4]]])
        corners_r = np.dot(rotate_matrix, corners.T)

        bounding_box_warp = np.full(5, -1, dtype=np.float32)
        bounding_box_warp[0] = bounding_box[0]
        bounding_box_warp[1:] = np.array([*np.min(corners_r, axis=1)-corners_min, *np.max(corners_r, axis=1)-corners_min]) #(class,x_min,y_min,x_max,y_max)
        bounding_boxes_warp.append(bounding_box_warp)
    bounding_boxes_warp = np.array(bounding_boxes_warp)

    M = np.zeros((2,3))
    M[:2,:2] = rotate_matrix
    M[0,2] = -corners_min[0]
    M[1,2] = -corners_min[1]

    img_warp = cv2.warpAffine(img, M, tuple(np.ceil(corners_max-corners_min).astype(np.int16)))

    return img_warp, bounding_boxes_warp


def zoom_out(img_warp, bounding_boxes_warp, img_bb_max_ratio=10):
    '''
    img_bb_max_ratio -> after zoom_out, every object must larger than 1/img_bb_max_ratio.


    '''
    zoom_max = img_bb_max_ratio

    h, w = img_warp.shape[:2]
    # for bounding_box in bounding_boxes_warp:
    #     ratio = np.min((bounding_box[3:5] - bounding_box[1:3]) * img_bb_max_ratio / [w, h])
    #     if zoom_max > ratio:
    #         zoom_max = ratio
    # if zoom_max <= 1:
    #     return img_warp, bounding_boxes_warp
    zoom = np.random.uniform(1, zoom_max)

    img_zoom_size = np.round(np.array([h, w])*zoom).astype(np.uint16)
    img_warp_zoom = np.zeros([*img_zoom_size,3], dtype=np.uint8)
    img_warp_zoom[..., 0] = np.random.randint(0, 256)
    img_warp_zoom[..., 1] = np.random.randint(0, 256)
    img_warp_zoom[..., 2] = np.random.randint(0, 256)
    img_warp_zoom[(img_zoom_size[0]-h)//2:(img_zoom_size[0]+h)//2, (img_zoom_size[1]-w)//2:(img_zoom_size[1]+w)//2] = img_warp
    bounding_boxes_warp_zoom = bounding_boxes_warp.copy()
    for i in [1,3]:
        bounding_boxes_warp_zoom[:,i:i+2] += (img_zoom_size[[1, 0]] - [w, h])/2

    return img_warp_zoom, bounding_boxes_warp_zoom


def resize(img_warp_zoom, bounding_boxes_warp_zoom, resize_range=[0.5, 3]):
    resize_ratio = np.random.uniform(resize_range[0],resize_range[1])

    img_resize_size = np.round([img_warp_zoom.shape[1]*resize_ratio,img_warp_zoom.shape[0]*resize_ratio]).astype(np.uint16)
    img_warp_zoom_resize = cv2.resize(img_warp_zoom, tuple(img_resize_size))
    bounding_boxes_warp_zoom_resize = bounding_boxes_warp_zoom.copy()
    bounding_boxes_warp_zoom_resize[:,3:] += 1
    bounding_boxes_warp_zoom_resize[:,1:] *=  resize_ratio
    bounding_boxes_warp_zoom_resize[:,3:] -= 1
    return img_warp_zoom_resize, bounding_boxes_warp_zoom_resize


def crop(img_warp_zoom_resize, bounding_boxes_warp_zoom_resize):
    bounding_boxes_warp_zoom_resize = np.round(bounding_boxes_warp_zoom_resize).astype(np.uint16)
    h, w = img_warp_zoom_resize.shape[:2]
    crop_ltrb = np.array([0, 0, w-1, h-1], dtype=np.uint16)   #left, top, right, bottom
    boundary_ltrb = np.zeros(4, dtype=np.uint16)
    boundary_ltrb[:2] = np.min(bounding_boxes_warp_zoom_resize[:,1:3], axis=0)
    boundary_ltrb[2:] = np.max(bounding_boxes_warp_zoom_resize[:,3:], axis=0)

    crop = crop_ltrb.copy()
    for i in range(len(crop_ltrb)):
        if i<2:
            crop[i] = np.random.randint(crop_ltrb[i],boundary_ltrb[i]+1)
        else:
            crop[i] = np.random.randint(boundary_ltrb[i], crop_ltrb[i]+1)

    img_crop = img_warp_zoom_resize[crop[1]:crop[3], crop[0]:crop[2]]

    bounding_boxes_crop = bounding_boxes_warp_zoom_resize.copy()
    for i in [1,3]:
        bounding_boxes_crop[:,i:i+2] -= crop[:2]

    return img_crop, bounding_boxes_crop


def write_img_and_label(augmentation_root, augmentation_label_root, img_name, idx, img_crop, bounding_boxes_crop):
    augmentation_path = os.path.join(augmentation_root, img_name[:-4] + "_" + str(idx) + img_name[-4:])
    augmentation_l_path = os.path.join(augmentation_label_root, img_name[:-4] + "_" + str(idx) + ".txt")

    h ,w = img_crop.shape[:2]

    cv2.imwrite(augmentation_path, img_crop)
    with open(augmentation_l_path, "w") as f:
        for bounding_box in bounding_boxes_crop:
            bounding_box_yolo = [bounding_box[0],
                                 (bounding_box[1]+bounding_box[3])/2, (bounding_box[2]+bounding_box[4])/2,
                                 bounding_box[3]-bounding_box[1], bounding_box[4]-bounding_box[2]]
            for i in range(1, len(bounding_box_yolo)):
                if i % 2:
                    bounding_box_yolo[i] /= w
                else:
                    bounding_box_yolo[i] /= h
            for i in range(len(bounding_box_yolo)):
                f.write(str(bounding_box_yolo[i]))
                if i != 4:
                    f.write(" ")
                else:
                    f.write("\n")

def imshow(img):
    cv2.imshow("img", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def augmentation(img_root, label_root, augmentation_root, augmentation_label_root, total_augmentation_image):
    if not os.path.isdir(augmentation_root):
        os.mkdir(augmentation_root)
    if not os.path.isdir(augmentation_label_root):
        os.mkdir(augmentation_label_root)

    img_names = [img_name for img_name in os.listdir(img_root) if img_name.endswith(".jpg")]
    img_num = len(img_names)
    augment_num_per_img = np.ones(img_num, dtype=np.uint16) * total_augmentation_image//img_num
    augment_num_per_img[:total_augmentation_image % img_num] += 1

    count = 0
    complete_percentage = 0
    processing_percentage_print("Complete Percentage = "+ str(complete_percentage) + "%")
    sys.stdout.write("Complete Percentage = "+ str(complete_percentage) + "%")
    for i, img_name in enumerate(img_names):
        img_path = os.path.join(img_root, img_name)
        label_path = os.path.join(label_root, img_name.replace('.jpg', '.txt'))
        img, bounding_boxes = get_img_and_bounding_boxes(img_path, label_path)

        for idx in range(augment_num_per_img[i]):

            img_warp, bounding_boxes_warp = rotation(img, bounding_boxes)
            img_warp_zoom, bounding_boxes_warp_zoom = zoom_out(img_warp, bounding_boxes_warp)
            img_warp_zoom_resize, bounding_boxes_warp_zoom_resize = resize(img_warp_zoom, bounding_boxes_warp_zoom)
            img_crop, bounding_boxes_crop = crop(img_warp_zoom_resize, bounding_boxes_warp_zoom_resize)

            write_img_and_label(augmentation_root, augmentation_label_root, img_name, idx, img_crop, bounding_boxes_crop)

            count += 1
            if complete_percentage != count*100//total_augmentation_image:
                complete_percentage += 1
                print_str = "Complete Percentage = "+ str(complete_percentage)+ "%"
                processing_percentage_print(print_str)
                sys.stdout.flush()
                sys.stdout.write('\r' + print_str)
    processing_percentage_print("Complete !!!")


def img_clicked():
    root = filedialog.askdirectory(initialdir=os.getcwd(),title='Please select a directory')
    img_str.set(root)


def label_clicked():
    root = filedialog.askdirectory(initialdir=os.getcwd(), title='Please select a directory')

    label_str.set(root)


def augmentation_img_clicked():
    root = filedialog.askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    augmentation_img_str.set(root)


def augmentation_label_clicked():
    root = filedialog.askdirectory(initialdir=os.getcwd(), title='Please select a directory')
    augmentation_label_str.set(root)


def process_clicked():
    augmentation(img_str.get(), label_str.get(), augmentation_img_str.get(),
                                  augmentation_label_str.get(), augmentation_num.get())


def multithread_process_clicked():
    t = threading.Thread(target=process_clicked)
    t.start()


def verify_clicked():
    validation(augmentation_img_str.get(), augmentation_label_str.get())


def processing_percentage_print(percentage_str):
    processing_percentage_str.set(percentage_str)


def main():
    img_root = "./Images"
    label_root = "./Labels"
    augmentation_root = "./AugmentationImages"
    augmentation_label_root = "./AugmentationLabels"
    total_augmentation_image = 1000
    augmentation(img_root, label_root, augmentation_root, augmentation_label_root, total_augmentation_image)


def validation(augmentation_root, augmentation_label_root):
    for img_name in os.listdir(augmentation_root):
        img_path = os.path.join(augmentation_root,img_name)
        img = cv2.imread(img_path)
        label_name = os.path.splitext(img_path.replace("Images/", "Labels/"))[0] + ".txt"
        with open(label_name, "r") as f:
            for line in f:
                bounding_box = np.array(line.split()[1:], dtype=np.float32)
                x_min = int((bounding_box[0] - bounding_box[2] / 2) * img.shape[1])
                x_max = int((bounding_box[0] + bounding_box[2] / 2) * img.shape[1])
                y_min = int((bounding_box[1] - bounding_box[3] / 2) * img.shape[0])
                y_max = int((bounding_box[1] + bounding_box[3] / 2) * img.shape[0])
                cv2.rectangle(img, (x_min,y_min), (x_max, y_max), (0,255,0), 10)

        imshow(cv2.resize(img,(512,512)))


def rectangle(img, bounding_box): #(x_min, y_min, x_max, y_max)

    bounding_box_rec = np.round(bounding_box.copy()).astype(np.uint16)
    img_rec = cv2.rectangle(img.copy().astype(np.uint8), tuple(bounding_box_rec[1:3]), tuple(bounding_box_rec[3:]), (0,0,255), 10)
    return img_rec

class Augmentation(object):
    def __init__(self):
        self.img_str = tk.StringVar()
        self.label_str = tk.StringVar()
        self.augmentation_img_str = tk.StringVar()
        self.augmentation_label_str = tk.StringVar()
        self.augmentation_num = tk.IntVar()
        
        self.color_jitter_ck = tk.BooleanVar()
        self.hsv_jitter_chk = tk.BooleanVar()
        self.rotation_chk = tk.BooleanVar()        
        self.rotation_x_range = [tk.IntVar(), tk.IntVar()]
        self.rotation_y_range = [tk.IntVar(), tk.IntVar()]
        self.rotation_z_range = [tk.IntVar(), tk.IntVar()]
        self.zoom_out_chk = tk.BooleanVar()  
        self.zoom_out_ratio = tk.DoubleVar()
        self.resize_chk = tk.BooleanVar()        
        self.resize_ratio = [tk.DoubleVar(), tk.DoubleVar()]
        self.crop_chk = tk.BooleanVar()    
    def multithread_process_clicked(self):
        def process_clicked():
            self.augmentation()        
        t = threading.Thread(target=process_clicked)
        t.start()
        
    def augmentation(self):
        img_root = self.img_str.get()
        label_root = self.label_str.get()
        augmentation_root = self.augmentation_img_str.get()
        augmentation_label_root = self.augmentation_label_str.get()
        total_augmentation_image = self.augmentation_num.get()
        
        if not os.path.isdir(augmentation_root):
            os.mkdir(augmentation_root)
        if not os.path.isdir(augmentation_label_root):
            os.mkdir(augmentation_label_root)
    
        img_names = [img_name for img_name in os.listdir(img_root) if img_name.endswith(".jpg") or img_name.endswith(".bmp")]
        img_num = len(img_names)
        # augment_num_per_img = np.ones(img_num, dtype=np.uint16) * total_augmentation_image//img_num
        # augment_num_per_img[:total_augmentation_image % img_num] += 1
    
        count = 0
        complete_percentage = 0
        processing_percentage_print("Complete Percentage = "+ str(complete_percentage) + "%")
        sys.stdout.write("Complete Percentage = "+ str(complete_percentage) + "%")
        for i, img_name in enumerate(img_names):
            print(img_name)
            img_path = os.path.join(img_root, img_name)
            label_path = os.path.join(label_root, os.path.splitext(img_name)[0] + '.txt')
            img, bounding_boxes = get_img_and_bounding_boxes(img_path, label_path)
    
            for idx in range(total_augmentation_image):
                img_process = img.copy()
                bounding_boxes_process = bounding_boxes.copy()
                if self.color_jitter_ck.get():
                    img_process = color_jitter(img_process)
                if self.hsv_jitter_chk.get():
                    img_process = hsv_augmentation(img_process)
                if self.rotation_chk.get():
                    img_process, bounding_boxes_process = rotation(img_process, bounding_boxes_process, 
                                                         [self.rotation_x_range[0].get(), self.rotation_x_range[1].get()], 
                                                         [self.rotation_y_range[0].get(), self.rotation_y_range[1].get()], 
                                                         [self.rotation_z_range[0].get(), self.rotation_z_range[1].get()])
                if self.zoom_out_chk.get():
                    img_process, bounding_boxes_process = zoom_out(img_process, bounding_boxes_process, self.zoom_out_ratio.get())
                if self.resize_chk.get():
                    img_process, bounding_boxes_process = resize(img_process, bounding_boxes_process, [self.resize_ratio[0].get(), self.resize_ratio[1].get()])
                if self.crop_chk.get():
                    img_process, bounding_boxes_process = crop(img_process, bounding_boxes_process)
    
                write_img_and_label(augmentation_root, augmentation_label_root, img_name, idx, img_process, bounding_boxes_process)
    
                count += 1
                if complete_percentage != count*100//total_augmentation_image//img_num:
                    complete_percentage += 1
                    print_str = "Complete Percentage = "+ str(complete_percentage)+ "%"
                    processing_percentage_print(print_str)
                    sys.stdout.flush()
                    sys.stdout.write('\r' + print_str)
        processing_percentage_print("Complete !!!")
        
    def verify_clicked(self):
        def validation(augmentation_root, augmentation_label_root):
            for img_name in os.listdir(augmentation_root):
                img_path = os.path.join(augmentation_root,img_name)
                img = cv2.imread(img_path)
                label_name = os.path.splitext(img_path.replace("Images/", "Labels/"))[0] + ".txt"
                with open(label_name, "r") as f:
                    for line in f:
                        bounding_box = np.array(line.split()[1:], dtype=np.float32)
                        x_min = int((bounding_box[0] - bounding_box[2] / 2) * img.shape[1])
                        x_max = int((bounding_box[0] + bounding_box[2] / 2) * img.shape[1])
                        y_min = int((bounding_box[1] - bounding_box[3] / 2) * img.shape[0])
                        y_max = int((bounding_box[1] + bounding_box[3] / 2) * img.shape[0])
                        cv2.rectangle(img, (x_min,y_min), (x_max, y_max), (0,255,0), 1)
        
                imshow(cv2.resize(img,(512,512)))
        validation(self.augmentation_img_str.get(), self.augmentation_label_str.get())



if __name__ == '__main__':
    
    
    window = Tk()
    window.title("AumentationTool")
    window.geometry('1000x800')
    
    augmentation_obj = Augmentation()
    
    root = os.getcwd()
    img_lb = tk.Label(window, width=22, text="Image Directory")
    img_lb.place(x=20, y=40)
    img_str = augmentation_obj.img_str
    img_str.set('/media/nickwang/StorageDisk/BOVIA/LPR/Dataset/CarImage')
    img_entry = tk.Entry(window, width=100, textvariable=img_str)
    img_entry.place(x=170, y=40)
    img_btn = tk.Button(window, text="Select", command=img_clicked)
    img_btn.place(x=900, y=40)

    label_lb = tk.Label(window, width=22, text="Label Directory")
    label_lb.place(x=20, y=80)
    label_str = augmentation_obj.label_str
    label_str.set('/media/nickwang/StorageDisk/BOVIA/LPR/Dataset/CarImage')
    label_entry = tk.Entry(window, width=100, textvariable=label_str)
    label_entry.place(x=170, y=80)
    label_btn = tk.Button(window, text="Select", command=label_clicked)
    label_btn.place(x=900, y=80)

    augmentation_img_lb = tk.Label(window, width=32, text="Augmentation Image Directory")
    augmentation_img_lb.place(x=20, y=120)
    augmentation_img_str = augmentation_obj.augmentation_img_str
    augmentation_img_str.set(root+'/AugmentationImages')
    augmentation_img_entry = tk.Entry(window, width=100, textvariable=augmentation_img_str)
    augmentation_img_entry.place(x=170, y=120)
    augmentation_img_btn = tk.Button(window, text="Select", command=augmentation_img_clicked)
    augmentation_img_btn.place(x=900, y=120)

    augmentation_label_lb = tk.Label(window, width=32, text="Augmentation Label Directory")
    augmentation_label_lb.place(x=20, y=160)
    augmentation_label_str = augmentation_obj.augmentation_label_str
    augmentation_label_str.set(root+'/AugmentationLabels')
    augmentation_label_entry = tk.Entry(window, width=100, textvariable=augmentation_label_str)
    augmentation_label_entry.place(x=170, y=160)
    augmentation_label_btn = tk.Button(window, text="Select", command=augmentation_label_clicked)
    augmentation_label_btn.place(x=900, y=160)

    augmentation_num_lb = tk.Label(window, width=22, text="Augmentation Quantity")
    augmentation_num_lb.place(x=20, y=200)
    augmentation_num = augmentation_obj.augmentation_num
    augmentation_num.set(1000)
    augmentation_num_entry = tk.Entry(window, width=10, textvariable=augmentation_num)
    augmentation_num_entry.place(x=200, y=200)
    
    color_jitter_ck = augmentation_obj.color_jitter_ck
    color_jitter_ck.set(True)
    color_jitter_bt = tk.Checkbutton(window, var=color_jitter_ck)
    color_jitter_bt.place(x=20, y = 240)
    color_jitter_lb = tk.Label(window, width=22, text="------Color Jitter------")
    color_jitter_lb.place(x=60, y=240)    
    
    hsv_jitter_chk = augmentation_obj.hsv_jitter_chk
    hsv_jitter_chk.set(True)
    hsv_jitter_bt = tk.Checkbutton(window, var=hsv_jitter_chk)
    hsv_jitter_bt.place(x=20, y = 280)
    hsv_jitter_lb = tk.Label(window, width=22, text="------HSV Jitter------")
    hsv_jitter_lb.place(x=60, y=280)    

    rotation_chk = augmentation_obj.rotation_chk
    rotation_chk.set(True)
    rotation_bt = tk.Checkbutton(window, var=rotation_chk)
    rotation_bt.place(x=20, y = 320)
    rotation_lb = tk.Label(window, width=22, text="-------Rotation-------")
    rotation_lb.place(x=60, y=320)
    rotation_x_lb = tk.Label(window, width=0, text="Range of x's degree")
    rotation_x_lb.place(x=250, y=320)
    rotation_x_min = augmentation_obj.rotation_x_range[0]
    rotation_x_min.set(-50)
    rotation_x_min_entry = tk.Entry(window, width=10, textvariable=rotation_x_min)
    rotation_x_min_entry.place(x=400, y=320)
    rotation_x_max = augmentation_obj.rotation_x_range[1]
    rotation_x_max.set(50)
    rotation_x_max_entry = tk.Entry(window, width=10, textvariable=rotation_x_max)
    rotation_x_max_entry.place(x=500, y=320)
    
    rotation_y_lb = tk.Label(window, width=0, text="Range of y's degree")
    rotation_y_lb.place(x=250, y=360)
    rotation_y_min = augmentation_obj.rotation_y_range[0]
    rotation_y_min.set(-50)
    rotation_y_min_entry = tk.Entry(window, width=10, textvariable=rotation_y_min)
    rotation_y_min_entry.place(x=400, y=360)
    rotation_y_max = augmentation_obj.rotation_y_range[1]
    rotation_y_max.set(50)
    rotation_y_max_entry = tk.Entry(window, width=10, textvariable=rotation_y_max)
    rotation_y_max_entry.place(x=500, y=360)
    
    rotation_z_lb = tk.Label(window, width=0, text="Range of z's degree")
    rotation_z_lb.place(x=250, y=400)
    rotation_z_min = augmentation_obj.rotation_z_range[0]
    rotation_z_min.set(-60)
    rotation_z_min_entry = tk.Entry(window, width=10, textvariable=rotation_z_min)
    rotation_z_min_entry.place(x=400, y=400)
    rotation_z_max = augmentation_obj.rotation_z_range[1]
    rotation_z_max.set(60)
    rotation_z_max_entry = tk.Entry(window, width=10, textvariable=rotation_z_max)
    rotation_z_max_entry.place(x=500, y=400)
    
    zoom_out_chk = augmentation_obj.zoom_out_chk
    zoom_out_chk.set(True)
    zoom_out_bt = tk.Checkbutton(window, var=zoom_out_chk)
    zoom_out_bt.place(x=20, y = 440)
    zoom_out_lb = tk.Label(window, width=22, text="-------Zoom Out-------")
    zoom_out_lb.place(x=60, y=440)
    zoom_out_lb = tk.Label(window, width=0, text="Ratio")
    zoom_out_lb.place(x=250, y=440)
    zoom_out_ratio = augmentation_obj.zoom_out_ratio
    zoom_out_ratio.set(2)
    zoom_out_ratio_entry = tk.Entry(window, width=10, textvariable=zoom_out_ratio)
    zoom_out_ratio_entry.place(x=400, y=440)
    
    resize_chk = augmentation_obj.resize_chk
    resize_chk.set(True)
    resize_bt = tk.Checkbutton(window, var=resize_chk)
    resize_bt.place(x=20, y = 480)
    resize_lb = tk.Label(window, width=22, text="-------Resize-------")
    resize_lb.place(x=60, y=480)
    resize_lb = tk.Label(window, width=0, text="Range of resize ratio")
    resize_lb.place(x=250, y=480)
    resize_min = augmentation_obj.resize_ratio[0]
    resize_min.set(0.5)
    resize_min_entry = tk.Entry(window, width=10, textvariable=resize_min)
    resize_min_entry.place(x=400, y=480)
    resize_max = augmentation_obj.resize_ratio[1]
    resize_max.set(2)
    resize_max_entry = tk.Entry(window, width=10, textvariable=resize_max)
    resize_max_entry.place(x=500, y=480)

    crop_chk = augmentation_obj.crop_chk
    crop_chk.set(True)
    crop_bt = tk.Checkbutton(window, var=crop_chk)
    crop_bt.place(x=20, y = 520)
    crop_lb = tk.Label(window, width=22, text="------Random Crop------")
    crop_lb.place(x=60, y=520)    
    
    processing_percentage_str = tk.StringVar()
    processing_percentage_str.set("Idle")
    processing_percentage_lb = tk.Label(window, width=30, textvariable=processing_percentage_str)
    processing_percentage_lb.place(x=200, y=660)

    processing_btn = tk.Button(window, text="Process", command=augmentation_obj.multithread_process_clicked)
    processing_btn.place(x=500, y=640)

    verify_btn = tk.Button(window, text="Verify", command=augmentation_obj.verify_clicked)
    verify_btn.place(x=500, y=680)

    window.mainloop()

    # main()
    # augmentation_root = "./AugmentationImages"
    # augmentation_label_root = "./AugmentationLabels"
    # validation(augmentation_root, augmentation_label_root)
