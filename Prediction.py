import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

# Define the breed names
breeds = [
    'Abyssinian','Bengal' ,'Birman' , 'Bombay', 'British_Shorthair',
    'Egyptian_Mau', 'Maine_Coon', 'Persian', 'Ragdoll', 'Russian_Blue', 'Siamese','Sphynx' ,
    'american_bulldog', 'american_pit_bull_terrier', 'basset_hound', 'beagle', 'boxer',
    'chihuahua', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired', 'great_pyrenees',
    'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'miniature_pinscher', 'newfoundland',
    'pomeranian', 'pug', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 
    'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier'
]

# Load the trained model
model = tf.keras.models.load_model('pet_breed_recognition_model_final.h5')

# Load and preprocess the input image
img_path = 'test.jpg'
img = image.load_img(img_path, target_size=(299, 299))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

# Make predictions
predictions = model.predict(img_array)[0]

# Get the top 10 predictions
top_10_indices = np.argsort(predictions)[-10:][::-1]
top_10_breeds = [(breeds[i], predictions[i]) for i in top_10_indices]

# Print the top 10 results
print("Top 10 predicted breeds:")
for breed, confidence in top_10_breeds:
    print(f"{breed}: {confidence:.2f}")
