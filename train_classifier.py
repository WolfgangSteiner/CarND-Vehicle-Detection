import glob
from imageutils import *
import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os
import pickle
from extract_features import extract_features
from itertools import product
import random
from multiprocessing import Pool as ThreadPool
from functools import partial


class DataSource(object):
    def __init__(self, purge=False):
        self.image_dir_vehicles = "vehicles"
        self.image_dir_non_vehicles = "non-vehicles"

        if os.path.exists("train_data.pickle") and not purge:
            self.load_pickled_data()
        else:
            self.load_data()
            self.split_data()
            #self.save_pickled_data()


    def load_data(self):
        X_vehicles, y_vehicles = self.load_data_from_dir(self.image_dir_vehicles, label=1)
        X_nonvehicles, y_non_vehicles = self.load_data_from_dir(self.image_dir_non_vehicles, label=0)
        self.X = {}
        self.y = {}
        for size in X_vehicles:
            print(X_vehicles[size].shape, X_nonvehicles[size].shape)
            self.X[size] = np.concatenate((X_vehicles[size], X_nonvehicles[size]), axis=0)
            self.y[size] = np.concatenate((y_vehicles[size], y_non_vehicles[size]), axis=0)


    def split_data(self):
        self.X_train, self.y_train = {}, {}
        self.X_val, self.y_val = {}, {}
        for size in self.X:
            self.X_train[size], self.X_val[size], self.y_train[size], self.y_val[size] = train_test_split(self.X[size], self.y[size], test_size=0.2, random_state=42)


    def save_pickled_data(self):
        with open("train_data.pickle", "wb") as f:
            pickle.dump((self.X_train,self.y_train), f)
        with open("validation_data.pickle", "wb") as f:
            pickle.dump((self.X_val,self.y_val), f)


    def load_pickled_data(self):
        with open("train_data.pickle", "rb") as f:
            self.X_train, self.y_train = pickle.load(f)
        with open("validation_data.pickle", "rb") as f:
            self.X_val, self.y_val = pickle.load(f)


    def process_image(self, img_file_name, label):
            X = {64:[], 32:[], 16:[]}
            img = load_img(img_file_name)
            for window_size in X:
                if window_size == 16:
                    img = scale_img(img,0.5)
                for augmented_img in self.augment_image(img, window_size, label):
                    augmented_img_yuv = bgr2yuv(augmented_img)
                    X[window_size].append(extract_features(augmented_img_yuv, window_size))

            return X


    def load_data_from_dir(self, dir, label):
        print("Loading data from %s" % dir)
        X = {64:[], 32:[], 16:[]}
        y = {64:[], 32:[], 16:[]}
        img_file_names = glob.glob("%s/**/*.[pj][np]g" % dir)
        max = None
        if max is not None:
            random.shuffle(img_file_names)
            img_file_names = img_file_names[0:max]

        n = len(img_file_names)

        with ThreadPool(processes=8) as pool:
            for i, X_dict in enumerate(pool.imap_unordered(partial(self.process_image,label=label), img_file_names, chunksize=1)):
                Utils.progress_bar(i, n)
                for size in X:
                    X[size].extend(X_dict[size])

        for size in X:
            y[size] = np.ones(len(X[size])) * label
            X[size] = np.array(X[size])

        return X, y


    def flip_image(self, img, label):
        result = [img]
        result.append(img[:,::-1,:])
        if False and label == 0:
            result.append(img[::-1,:,:])
            result.append(img[::-1,::-1,:])

        return result


    def window_positions(self, img_size, window_size, delta=None):
        if delta == None:
            delta = max(1, img_size // 2)
        result = []
        x = 0
        while x <= img_size - window_size:
            result.append(x)
            x += delta

        return result


    def augment_image(self, img, window_size, label):
        w,h = img_size(img)
        if w == window_size:
            return self.flip_image(img, label)

        result = []
        positions = self.window_positions(w,window_size)
        for x,y in product(positions,positions):
            window = crop_img(img, x,y,x+window_size,y+window_size)
            result.extend(self.flip_image(window, label))

        result.extend(self.augment_image(scale_img(img, 0.5), window_size, label))
        return result


source = DataSource(purge=True)

svc = {64:None, 32:None, 16:None}
scaler = {64:None, 32:None, 16:None}
res = {}

def train_classifier(size):
    print("Size: %dx%d:" % (size, size))
    my_scaler = StandardScaler()
    print("Fitting scaler with input array of size: ", source.X_train[size].shape)
    my_scaler.fit(source.X_train[size])
    print(my_scaler.mean_.shape)
    my_svc = LinearSVC(verbose=False, dual=False)
    my_svc.fit(my_scaler.transform(source.X_train[size]), source.y_train[size])
    score = my_svc.score(my_scaler.transform(source.X_val[size]), source.y_val[size])
    print("Size: %d:" % size, score)
    return (my_scaler, my_svc)


with ThreadPool(processes=len(source.X_train.keys())) as pool:
    result = pool.map(train_classifier, source.X_train.keys())
    pool.close()
    pool.join()


for idx,size in enumerate(svc.keys()):
    scaler[size] = result[idx][0]
    svc[size] = result[idx][1]


with open("svc.pickle", "wb") as f:
    pickle.dump((svc,scaler), f)
