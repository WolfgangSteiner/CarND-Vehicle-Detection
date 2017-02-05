import cv2
import numpy as np
import glob
import pickle
import os
import cv2grid
import imageutils

def calibrate_camera():
    """
    Calculate the camera matrix and ditortion coefficients. For this,
    the calibration images from the camear_cal folder are analyzed.
    """

    object_points = []
    image_points = []

    # read calibration images
    for f in glob.glob("camera_cal/calibration*.jpg"):
        print("Processing {}".format(f))
        # load image and convert to grayscale:
        img = imageutils.load_img(f)
        img = imageutils.bgr2gray(img)

        # detect chessboard corners
        pattern_found, corners = cv2.findChessboardCorners(img, (9,6))

        # if chessboard was detected: add object and image points
        if pattern_found:
            op = np.zeros((6*9,3), np.float32)
            op[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
            object_points.append(op)
            image_points.append(corners)

    # use the detected chessboard points to calculate the camera matrix and
    # the distortoin_coefficients
    camera_matrix = np.zeros((3,3))
    distortion_coefficients = np.zeros(5)
    cv2.calibrateCamera(object_points, image_points, img.shape[0:2], camera_matrix, distortion_coefficients)

    return camera_matrix, distortion_coefficients

# this code is executed when the module is imported
# if distortion coefficients are already pickled: load pickled data
if os.path.exists("camera_calibration.pickle"):
    with open("camera_calibration.pickle", "rb") as f:
        camera_matrix = pickle.load(f)
        distortion_coefficients = pickle.load(f)
# otherwise: calculate calibration data and save to pickle.
else:
    camera_matrix, distortion_coefficients = calibrate_camera()
    with open("camera_calibration.pickle", "wb") as f:
        pickle.dump(camera_matrix, f)
        pickle.dump(distortion_coefficients, f)


def undistort_image(img):
    """
    Undistort an image using the calculated distortion coefficients and
    camera matrix.
    """
    return cv2.undistort(img, camera_matrix, distortion_coefficients)


# plot calibration data when called as main:
if __name__ == '__main__':
    c = cv2grid.CV2Grid(1280/2,720/4*2,grid=(2,2))
    c.draw_grid()
    for i,img_name in enumerate(("camera_cal/calibration1.jpg", "test_images/straight_lines1.jpg")):
        print(img_name)
        img = imageutils.load_img(img_name)
        udist_img = undistort_image(img)
        c.paste_img(img, (0,i),scale=0.25,title="original",title_style="topcenter")
        c.paste_img(udist_img, (1,i),scale=0.25,title="distortion corrected",title_style="topcenter")
    c.save("output_images/camera_calibration.png")
