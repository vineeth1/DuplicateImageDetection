import os
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

image_folder = os.getcwd() + '/Images'
images = [i for i in os.listdir(image_folder)]
num_of_images = len(images)
vgg_vectors = []
loaded_images = []

# finding the vgg_vectors for the given set of images
for img in images:
    img_path = image_folder + '/' + img
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    
    vgg16_feature = model.predict(img_data)
    
    loaded_images.append(img)
    vgg_vectors.append(vgg16_feature.flatten())

query_folder = os.getcwd() + '/Queries'
queries = [i for i in os.listdir(query_folder)]

for query in queries:
    img_path = query_folder + '/' + query
    
    # finding vgg_vector of the query image    
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    query_vgg_vector = model.predict(img_data).flatten()
    
    # euclidean distance of query image's vgg vector to all other vgg vectors
    dist_vector = [np.linalg.norm(query_vgg_vector - vgg_vector) for vgg_vector in vgg_vectors]
    
    # sorting the images w.r.t euclidean distance from query image
    sorted_images = sorted(range(num_of_images), key=lambda k: dist_vector[k])
    
    # nearest image index
    duplicate_img_index = sorted_images[0]
        
    print('query image = ' + query)
    print('duplicate image = ' + images[duplicate_img_index] + '\n')
