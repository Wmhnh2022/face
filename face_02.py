import os
import cv2
import time
import sqlite3
import pathlib
import numpy as np
from tkinter import *
from PIL import Image, ImageTk, ImageFont, ImageDraw

name = ''  # 当前人脸的名字
lastid = 0  # 用户最新的id
id_name_map = {}  # 用户id对应名字
name_id_map = {}  # 用户名对应id


def cv2_putChinese(image, chinese, xy, font='msyh.ttc', size=25, fill=(255, 0, 0)):


    """cv2转PIL绘制中文后转回cv2"""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    font = ImageFont.truetype(font, size)
    draw = ImageDraw.Draw(image)
    draw.text(xy, chinese, font=font, fill=fill)#文本
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image


def show_on_tkinter(image, title):


    """用tkinter显示图片"""


    def save(event):


        global name
        global lastid
        name = entry.get()
        if name:
            if name not in name_id_map:
                c.execute('INSERT INTO users (`name`) VALUES (?)', (name,))  # 插入数据库
                conn.commit()  # 提交
                lastid += 1  # 更新用户最新的id
                id_name_map[lastid] = name
                name_id_map[name] = lastid  # 更新所有用户
                if name_id_map:
                    print('数据库中的用户有: {}'.format(' '.join(name_id_map)))  # 所有用户
            os.makedirs('dataset/{}'.format(name), exist_ok=True)  # 保存人脸图像目录
            filename = 'dataset/{}/{}.jpg'.format(name, int(time.time()))  # 保存人脸图像文件名
            image.save(filename)  # 用Image.save()避免cv2.imwrite()不能中文名的缺点
        window.destroy()

    window = Tk()
    window.title(title)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = image.shape
    canvas = Canvas(window, width=width, height=height)
    canvas.pack()
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, image=photo, anchor=NW)
    label = Label(window, text='输入姓名，空则跳过')
    label.pack(anchor=CENTER)
    entry = Entry(window)
    entry.pack(anchor=CENTER)
    entry.focus_force()
    entry.bind('<Return>', func=save)
    window.mainloop()

# 人脸数据库
conn = sqlite3.connect('database.db')  # 人脸数据库
c = conn.cursor()#创建游标
sql = '''
CREATE TABLE IF NOT EXISTS users (
`id` INTEGER UNIQUE PRIMARY KEY AUTOINCREMENT,
`name` TEXT UNIQUE
);
'''
c.execute(sql)  # 用户表
users = c.execute('SELECT * FROM users')

for (id, name) in users:
    lastid = id
    id_name_map[lastid] = name
    name_id_map[name] = id
if name_id_map:
    print('数据库中的用户有: {}'.format(' '.join(name_id_map)))  # 所有用户

# 记录人脸
os.makedirs('dataset', exist_ok=True)  # 保存人脸图像目录
path='E:\\opencv_4\\facelook\\haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(path)  # 加载模型
images = pathlib.Path('image').rglob('*')
for image in images:
    print('正在处理: {}'.format(image))
    image = str(image)
    image = cv2.imread(image)
    original = image.copy()
    cv2.imshow('original', original)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    # faces = model.detectMultiScale(gray)
    faces = model.detectMultiScale(gray)
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y:y + h, x:x + w]#剪切人脸部分
        cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv2.imshow('original', original)
        show_on_tkinter(face, title=i + 1)
cv2.destroyAllWindows()
conn.close()

# 训练人脸识别器
recognizer = cv2.face.LBPHFaceRecognizer_create()
ids = []
faces = []
for name in pathlib.Path('dataset').rglob('*'):
    images = pathlib.Path(name).glob('*')
    for image in images:
        ids.append(name_id_map[name.name])
        image = Image.open(image).convert('L')
        image = np.array(image)
        faces.append(image)
ids = np.array(ids)
recognizer.train(faces, ids)
recognizer.save('recognizer.yml')  # 保存人脸识别器模型

# 使用人脸识别器

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('recognizer.yml')  # 加载人脸识别器
# img = cv2.imread('image/2.webp')  # 选一张图片识别
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    cap.release()  # 释放视频
    sys.exit('sorry')
while cap.isOpened():
    ret, frame = cap.read()
    if ret == False:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = model.detectMultiScale(gray, 1.1, 5, cv2.CASCADE_SCALE_IMAGE, (50, 50))
    # faces = model.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        ids, conf = recognizer.predict(gray[y:y + h, x:x + w])
        name = id_name_map[ids]
        print(ids, name, conf)
        frame = cv2_putChinese(frame, name, (x + 2, y + h - 5))
    cv2.imshow('result', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
# cv2.imwrite('result.jpg', img)
# print('已保存')
