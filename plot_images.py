import glob
from sys import argv

from cv2grid import CV2Grid

image_dir = argv[1]
img_names = glob.glob("%s/*.png" % image_dir)
img_size = (64, 64)
num_cols = 30
num_rows = len(img_names) // num_cols + (len(img_names) % num_cols > 0)

c = CV2Grid((num_cols * 64, num_rows * 64), (num_cols, num_rows))

for i, img_name in enumerate(img_names):
    c.paste_img_file(img_name, (i % num_cols, i // num_cols))

c.save("%s/overview.png" % argv[1])
