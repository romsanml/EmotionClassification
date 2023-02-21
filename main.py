import face_recognition
from PIL import Image, ImageDraw
from emotion import Emotion
import numpy as np

full_image = face_recognition.load_image_file('Surprice.jpeg')

face_locations = face_recognition.face_locations(full_image)

general_image = Image.fromarray(full_image)

draw = ImageDraw.Draw(general_image)

emotion = Emotion()
emotion.fit()

for i, face_location in enumerate(face_locations):
    top, right, bottom, left = face_location
    face_image = full_image[top:bottom, left:right]
    face_image = Image.fromarray(face_image).convert('L')
    gray_image = np.asarray(face_image)
    gray_image = np.expand_dims(gray_image, axis=2)

    name = emotion.predict(gray_image)

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, top - text_height - 10), (right, bottom)), outline=(0, 0, 255))
    draw.rectangle(((left, top - text_height - 10), (right, top)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, top - text_height - 5), name, fill=(255, 255, 255, 255))

general_image.show()

general_image.save('image_with_boxes.jpg')
