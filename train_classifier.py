import glob
from imageutils import *
import Utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os
import pickle
from extract_features import extract_features


class DataSource(object):
    def __init__(self, purge=False):
        self.image_dir_vehicles = "vehicles"
        self.image_dir_non_vehicles = "non-vehicles"

        if os.path.exists("train_data.pickle") and not purge:
            self.load_pickled_data()
        else:
            self.load_data()
            self.split_data()
            self.save_pickled_data()


    def load_data(self):
        X_vehicles, y_vehicles = self.load_data_from_dir(self.image_dir_vehicles, label=1)
        X_nonvehicles, y_non_vehicles = self.load_data_from_dir(self.image_dir_non_vehicles, label=0)
        self.X = {}
        for size in X_vehicles:
            print(X_vehicles[size].shape, X_nonvehicles[size].shape)
            self.X[size] = np.concatenate((X_vehicles[size], X_nonvehicles[size]), axis=0)

        self.y = np.concatenate((y_vehicles, y_non_vehicles), axis=0)


    def split_data(self):
        self.X_train = {}
        self.X_val = {}
        for size in self.X:
            self.X_train[size], self.X_val[size], self.y_train, self.y_val = train_test_split(self.X[size], self.y, test_size=0.2, random_state=42)


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



    def load_data_from_dir(self, dir, label):
        print("Loading data from %s" % dir)
        X = {64:[], 32:[], 16:[]}
        img_file_names = glob.glob("%s/**/*.[pj][np]g" % dir)
        n = len(img_file_names)
        for idx,f in enumerate(img_file_names):
            Utils.progress_bar(idx+1,n)
            img = load_img(f)

            for img in self.augment_image(img, label):
                for size,array in X.items():
                    scaled_img = scale_img(img, size / 64)
                    array.append(extract_features(scaled_img, size))

        y = np.zeros(len(X[64]))
        y[:] = label

        for size in X:
            X[size] = np.array(X[size])

        return X, y


    def augment_image(self, img, label):
        result = [img]
        result.append(img[:,::-1,:])
        if label == 0:
            result.append(img[::-1,:,:])
            result.append(img[::-1,::-1,:])

        return result


source = DataSource(purge=True)

svc = {}
scaler = {}
for size in source.X_train:
    print("Size: %dx%d:" % (size,size))
    scaler[size] = StandardScaler()
    scaler[size].fit(source.X_train[size])
    svc[size] = LinearSVC()
    svc[size].fit(scaler[size].transform(source.X_train[size]), source.y_train)
    print(svc[size].score(scaler[size].transform(source.X_val[size]),source.y_val))

with open("svc.pickle", "wb") as f:
    pickle.dump((svc,scaler), f)
