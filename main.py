import cv2
from pylibdmtx.pylibdmtx import decode
import os
from multiprocessing import Pool
from timeit import default_timer as timer
from functools import partial
import numpy as np


def dmtx_reader(img):
    return decode(img)


def get_path_list(path):
    list_path = []
    for i in os.walk(path):
        if i[0] != path:
            for j in i[2]:
                list_path.append(i[0] + '/' + j)
    return list_path


def contrast(img):
    clahe = cv2.createCLAHE(clipLimit=5., tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def refactor(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(img_grey, cv2.CV_64F).var()
    print(fm)

    if fm < 50:

        sharp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharp_img = cv2.filter2D(img_grey, ddepth=-1, kernel=sharp_filter)
        return sharp_img

    if fm > 500:

        blr_img = cv2.medianBlur(img, 3)
        morph_kernel = np.ones((3, 3))
        erode_img = cv2.dilate(blr_img, kernel= morph_kernel, iterations=1)
        return erode_img

    return img


def draw_dmtx_data_save(path, path_save):
    img = cv2.imread(path)
    h, w = img.shape[:2]
    if h >= 2000 or w >= 2000:
        img = cv2.resize(img, (w // 2, h // 2))  # слишком долго без этого, но на качество, конечно, отрицательно влияет

    img_new = refactor(img)
    flag = False
    data = dmtx_reader(img_new)

    if len(data):
        with open(f"{path_save}opend.txt", 'a') as p:
            p.write(f"{path} DATA: {data}\n")
        print(path, data)
        img = cv2.flip(img, 0)
        left = data[0].rect.left
        top = data[0].rect.top
        height = data[0].rect.height
        width = data[0].rect.width
        # не всегда правильно их рисует
        img = cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), thickness=3)
        flag = True
    if flag:
        cv2.imwrite(f"{path_save}detect/{path.split('/')[-2]}_{path.split('/')[-1]}", img)
        cv2.imwrite(f"{path_save}detect/{path.split('/')[-2]}_{path.split('/')[-1]}_refact", img_new)
    else:
        cv2.imwrite(f"{path_save}no_detect/{path.split('/')[-2]}_{path.split('/')[-1]}", img)
        cv2.imwrite(f"{path_save}no_detect/{path.split('/')[-2]}_{path.split('/')[-1]}_refact", img_new)
    return 1 if flag else 0


if __name__ == '__main__':
    # тестировал на этих избражениях так как там их больше всего и там прдставлены разные ситуации
    PATH = "/home/pashnya/Документы/qr_data/2022_08_14/"

    PATH_SAVE = f"{os.getcwd()}/test/"

    if not os.path.exists(PATH_SAVE):
        os.mkdir(PATH_SAVE)
    if not os.path.exists(f"{PATH_SAVE}detect/"):
        os.mkdir(f"{PATH_SAVE}detect/")
    if not os.path.exists(f"{PATH_SAVE}no_detect/"):
        os.mkdir(f"{PATH_SAVE}no_detect/")

    path_list = get_path_list(PATH)
    with open(f"{PATH_SAVE}opend.txt", 'w') as p:
        p.write("")

    start_time = timer()
    with Pool(4) as p:
        count_detected = sum(p.map(partial(draw_dmtx_data_save, path_save=PATH_SAVE), path_list))
    print(f"time: {timer() - start_time}")
    print(f"count: {count_detected}")
