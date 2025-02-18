import streamlit as st
from function import get_photo, send_results
from function import DATA_PATH, DEVICE, TRANSFORM
from ultralytics import YOLO

import torch
from PIL import Image
import cv2

det_model = YOLO('./runs/detect/face_detection/weights/best.pt')
rec_model = torch.load('face_recognition.pth', weights_only=False)
rec_model.eval()



st.title('Распознавание человека по фотографии')

uploader = st.file_uploader('Выберите фотографию', ['png', 'jpeg', 'jpg'])

if uploader:
    user_img = 'user_img.jpg'
    print(uploader.name)
    # созраняем фото во временный файл
    with open(user_img, 'wb') as f:
        f.write(uploader.getbuffer())
        crops = get_photo(user_img, det_model, TRANSFORM, DEVICE)
        for i in range(len(crops)):
            crop = crops[i]
            # test = tests[i]
            # print(type(test))

            image = Image.fromarray(crop)
            person, conf = send_results(image_data=crop, 
                                        rec_model=rec_model,
                                        train_path=DATA_PATH,
                                        transform=TRANSFORM)
            if conf < 0.5:
                st.image(image, caption=f'Uknown person')
            else:
                st.image(image, caption=f'{person} ({conf:.3f})')
        st.image(Image.open(user_img))

