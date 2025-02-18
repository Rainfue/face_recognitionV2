#
from PIL import Image
import cv2

#
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision.datasets import ImageFolder

# библиотека для массивов
import numpy as np

#
from ultralytics import YOLO



# конфигурация
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH = './Data/face_recognition/train'
# пайплайн для обработки фотографий
TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),   # изменение размера на 224х224
    transforms.ToTensor(),          # приведение к тензору
    transforms.Normalize([0.5],[0.5]) # нормализация (mean=0.5, std=0.5)
])



# функция для получения фото
def get_photo(img_path: str, model_det: YOLO, transform, device):
    res = model_det.predict(img_path, conf=0.3, iou=0.2)
    image = cv2.imread(img_path)
    crops = []
    # проходимся по результатам детекции
    for result in res:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1,y1,x2,y2 = map(int, box)
            crop = image[y1:y2, x1:x2]
            print('1', type(crop))

            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            print('2', type(crop))

            # test = Image.fromarray(crop)
            # print('3', type(test))

            # test = transform(test).unsqueeze(0).to(DEVICE)
            # print('4', type(test))
            crops.append(crop)
            # tests.append(test)

    print('Найдено лиц:', len(crops))

    return crops


# функция распознавания
def send_results(image_data: str | np.ndarray, rec_model, train_path=DATA_PATH, transform=TRANSFORM):

    train_dataset = ImageFolder(root=train_path)
    print(len(train_dataset.classes))
    print(train_dataset.classes)

    if type(image_data) == str:
        image = Image.open(image_data).convert('RGB')
        image = transform(image).unsqueeze(0).to(DEVICE)
        print(type(image))
        
    if type(image_data) == np.ndarray:
        image = Image.fromarray(image_data)
        print(type(image))
        image = transform(image).unsqueeze(0).to(DEVICE)
        print(type(image))
        print(image.shape)
        
    with torch.no_grad():
        output = rec_model(image)
        proba = F.softmax(output, dim=1)
        print(proba)
        predict = torch.argmax(proba, dim=1).item()
        print(f'predict {predict}')
 
    class_name = train_dataset.classes[predict]
    print(class_name, proba.max().item())
    return class_name, proba.max().item()