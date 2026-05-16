import tensorflow as tf
import numpy as np
import sys

# Load model
model = tf.keras.models.load_model('tomato_model.h5')
print("Model loaded OK")

# Load a real image from the uploads folder
import os
uploads = os.listdir('uploads')
if not uploads:
    print("ERROR: No images in uploads folder yet.")
    print("Run the app, upload an image first, then run this script.")
    sys.exit()

img_path = os.path.join('uploads', uploads[0])
print(f"Testing with: {img_path}")

img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
arr = tf.keras.preprocessing.image.img_to_array(img)
arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
arr = np.expand_dims(arr, axis=0)

pred = model.predict(arr)[0]

classes = [
    'Tomato_Bacterial_Spot',
    'Tomato_Early_Blight',
    'Tomato_Late_Blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_Leaf_Spot',
    'Tomato_Spider_Mites',
    'Tomato_Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus',
    'Tomato_healthy',
]

print()
print('All predictions:')
for i, (cls, p) in enumerate(zip(classes, pred)):
    bar = '█' * int(p * 40)
    print(f'  {i}: {cls:<40} {p*100:5.1f}%  {bar}')

print()
print('Top result:', classes[pred.argmax()], f'({pred.max()*100:.1f}%)')