# yolo_01.py
# pydot install 필요 pip install pydot
import tensorflow
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt


# 모델 파일 로딩
# model = load_model('model_data\\yolo.h5')
# model.summary()
# 구조를 이미지로 저장
# plot_model(model, show_shapes=True, to_file='yolo3_structure.png')
# plt.imshow('yolo3_structure.png')
# plt.show()
from keras_preprocessing.image import load_img

# 이미지가 있는 path를 아래에 적음
# path = "D:\\PersonalData\\Desktop\\AI\\Project\\YOLO\\"
# filename = 'street_view.jpg'
path = "C:\\Users\\user\\Desktop\\"
filename = 'graduation.jpg'
img = plt.imread(path + filename)
plt.imshow(img)
# plt.show()

import keras.backend as K
# print(tensorflow.compat.v1.Session())


from yolo import YOLO
def do_object_detection(file, modelPath, class_path):
    yolo = YOLO(model_path=modelPath, classes_path=class_path)
    # 이미지 로딩
    image = load_img(file, target_size=(416, 416))
    # 실행 : detect함
    result_image = yolo.detect_image(image)
    # 실행 결과 표시
    plt.imshow(result_image)
    plt.show()

do_object_detection(path + filename, '.\\model_data\\yolo.h5', '.\\model_data\\coco_classes.txt')

