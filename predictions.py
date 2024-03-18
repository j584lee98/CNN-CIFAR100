import numpy as np
from keras.models import load_model
from keras.utils import image_utils

cifar_model = load_model('cifar-100.h5')
str_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def format_list(str_list):
    upper = [x.capitalize() for x in str_list]
    split = [" ".join(x.split('_')) for x in upper]
    return split

def image_predict(model, image):
    img = image_utils.load_img(image, target_size=(32, 32, 3))
    img = image_utils.img_to_array(img)
    img = img/255
    img = np.expand_dims(img, axis=0)
    
    model_pred = model.predict(img)[0]
    preds = dict(zip(format_list(str_labels), model_pred))
    preds_sorted = sorted(preds.items(), key=lambda x:x[1], reverse=True)

    return preds_sorted[:3]