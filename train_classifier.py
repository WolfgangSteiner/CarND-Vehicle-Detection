import glob
import os
import pickle
import random
from functools import partial
from itertools import product
from multiprocessing import Pool as ThreadPool
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

import xgboost as xgb
import Utils
from extract_features import extract_features
from imageutils import *

hires = False
type = "svc"

class DataSource(object):
    def __init__(self, purge=False):
        self.image_dirs_vehicles = ("vehicles",)
        self.image_dirs_non_vehicles = ("non-vehicles", "false_positives*")

        if hires:
            self.sizes = (64,)
        else:
            self.sizes = (64, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16)

        if os.path.exists("train_data.pickle") and not purge:
            self.load_pickled_data()
        else:
            self.load_data()
            self.split_data()
            #self.save_pickled_data()


    def empty_dict(self):
        d = {}
        for size in self.sizes:
            d[size] = []
        return d


    def load_data(self):
        X_vehicles, y_vehicles = self.load_data_from_dirs(self.image_dirs_vehicles, label=1)
        X_nonvehicles, y_non_vehicles = self.load_data_from_dirs(self.image_dirs_non_vehicles, label=0)
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
            X = self.empty_dict()
            for size in self.sizes:
                X[size] = []
            img = load_img(img_file_name)
            w,h = img_size(img)
            for window_size in X:
                if w < window_size:
                    continue
                elif False and label == 0 and window_size < w and "Extras" in img_file_name:
                    scaled_img = scale_img(img,window_size / w * 2)
                else:
                    scaled_img = scale_img(img,window_size / w)


                ppc = 8 if hires else 16
                for augmented_img in self.augment_image(scaled_img, window_size, label):
                    augmented_img_yuv = bgr2yuv(augmented_img)
                    X[window_size].append(extract_features(augmented_img_yuv, window_size, ppc=ppc))

            return X


    def load_data_from_dirs(self, dirs, label):
        img_file_list = []

        for dir in dirs:
            img_file_list.extend(glob.glob("%s/**/*.[pj][np]g" % dir))

        return self.load_data_from_file_list(img_file_list, label)


    def load_data_from_file_list(self, img_file_list, label):
        X = self.empty_dict()
        y = self.empty_dict()
        max = None
        if max is not None:
            random.shuffle(img_file_list)
            img_file_list = img_file_list[0:max]

        n = len(img_file_list)

        with ThreadPool(processes=8) as pool:
            for i, X_dict in enumerate(pool.imap_unordered(partial(self.process_image,label=label), img_file_list, chunksize=1)):
                Utils.progress_bar(i, n)
                for size in X:
                    X[size].extend(X_dict[size])

        for size in X:
            if label == 1:
                y[size] = np.ones(len(X[size]))
            else:
                y[size] = np.zeros(len(X[size]))

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
classifier = source.empty_dict()
scaler = source.empty_dict()
res = {}

def train_svc(size):
    my_scaler = StandardScaler()
    my_scaler.fit(source.X_train[size])
    my_svc = LinearSVC()
    X_train = my_scaler.transform(source.X_train[size])
    my_svc.fit(X_train, source.y_train[size])
    #my_svc_sigmoid = CalibratedClassifierCV(my_svc, cv=2, method='sigmoid')
    #my_svc_sigmoid.fit(X_train, source.y_train[size])
    score = my_svc.score(my_scaler.transform(source.X_val[size]), source.y_val[size])
    print("Size: %d:" % size, score)
    return (my_scaler, my_svc)


def train_xgb(size):
    my_scaler = StandardScaler()
    my_scaler.fit(source.X_train[size])
    my_xgb = xgb.XGBClassifier()
    X_train = my_scaler.transform(source.X_train[size])
    my_xgb.fit(X_train, source.y_train[size])
    score = my_xgb.score(my_scaler.transform(source.X_val[size]), source.y_val[size])
    print("Size: %d:" % size, score)
    return (my_scaler, my_xgb)


train_func = train_xgb if type=="xgb" else train_svc

with ThreadPool(processes=min(8, len(source.sizes))) as pool:
    result = pool.map(train_func, source.X_train.keys())
    pool.close()
    pool.join()


for idx,size in enumerate(classifier.keys()):
    scaler[size] = result[idx][0]
    classifier[size] = result[idx][1]

suffix = "hires" if hires else "multires"

with open("%s_%s.pickle" % (type,suffix), "wb") as f:
    pickle.dump((classifier,scaler), f)
