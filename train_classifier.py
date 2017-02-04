import glob
from imageutils import *
import Utils
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import os
import pickle

class DataSource(object):
    def __init__(self, purge=False):
        self.image_dir_vehicles = "vehicles/KITTI_extracted"
        self.image_dir_non_vehicles = "non-vehicles/Extras"

        if os.path.exists("train_data.pickle") and not purge:
            self.load_pickled_data()
        else:
            self.load_data()
            self.split_data()
            self.save_pickled_data()


    def load_data(self):
        X_vehicles, y_vehicles = self.load_data_from_dir(self.image_dir_vehicles, label=1)
        X_non_vehicles, y_non_vehicles = self.load_data_from_dir(self.image_dir_non_vehicles, label=0)
        print(X_vehicles.shape, X_non_vehicles.shape)
        self.X = np.concatenate((X_vehicles, X_non_vehicles), axis=0)
        self.y = np.concatenate((y_vehicles, y_non_vehicles), axis=0)


    def split_data(self):
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


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
        X = []
        img_file_names = glob.glob("%s/*.[pj][np]g" % dir)
        n = len(img_file_names)
        for idx,f in enumerate(img_file_names):
            Utils.progress_bar(idx+1,n)
            img = load_img(f)

            for img in self.augment_image(img, label):
                img_y,_,_ = split_yuv(img)
                hog_data = hog(img_y, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                                visualise=False, transform_sqrt=False, feature_vector=True,
                                normalise=None)
                X.append(hog_data)

        y = np.zeros(len(X))
        y[:] = label
        return np.array(X), y


    def augment_image(self, img, label):
        result = [img]
        result.append(img[:,::-1,:])
        if label == 0:
            result.append(img[::-1,:,:])
            result.append(img[::-1,::-1,:])

        return result


source = DataSource(purge=True)

svc = LinearSVC()
svc.fit(source.X_train, source.y_train)
print(svc.score(source.X_val,source.y_val))