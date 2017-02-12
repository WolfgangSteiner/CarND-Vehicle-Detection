import cv2
from ImageThresholding import *
from imageutils import *
from midicontrol import MidiControlManager, MidiControl

class FilterPipeline(MidiControlManager):
    def __init__(self):
        super().__init__()
        self.intermediates = []

    def process(self,img):
        img = do_process(img)
        return img


    def do_process(self,img):
        return img


    def add_intermediate(self, img, title=None):
        self.intermediates.append((img,title))

    def add_intermediate_channel(self, channel, title=None):
        self.intermediates.append((expand_channel(channel),title))

    def add_intermediate_mask(self, mask, title=None):
        self.intermediates.append((expand_mask(mask),title))

    def add_empty_intermediate(self):
        self.intermediates.append((None,None))


class YUVPipeline(FilterPipeline):
    def __init__(self):
        super().__init__()
        self.y_y_min = MidiControl(self, "y_y_min", 70, value=116)
        self.y_u_min = MidiControl(self, "y_u_min", 71, value=240)
        self.y_v_max = MidiControl(self, "y_v_max", 72, value=32)

        self.w_y_min = MidiControl(self, "w_y_min", 73, value=245)
        self.w_uv_max = MidiControl(self, "w_uv_max", 105, value=32)

        self.mag_y_min = MidiControl(self, "mag_y_min", 87, value=20)
        self.mag_y_max = MidiControl(self, "mag_y_max", 111, value=255)
        self.mag_y_ksize = MidiControl(self, "mag_y_ksize", 106, value=5, allowed_values=range(3,33,2))

        self.dir_y_dir = MidiControl(self, "dir_y_dir", 88, value=1.0, min=0.0, max=2.0)
        self.dir_y_delta = MidiControl(self, "dir_y_delta", 112, value=0.5, min=0.0, max=1.0)
        self.dir_y_ksize = MidiControl(self, "dir_y_ksize", 107, value=5, allowed_values=range(3,33,2))

        self.mag_v_min = MidiControl(self, "mag_v_min", 7, value=16)
        self.mag_v_max = MidiControl(self, "mag_v_max", 116, value=255)
        self.mag_v_ksize = MidiControl(self, "mag_v_ksize", 108, value=7, allowed_values=range(3,33,2))

        self.eq_limit = MidiControl(self, "eq_limit", 80, value=0.2, min=0.005, max=1.0)
        self.eq_ny = MidiControl(self, "eq_ny", 81, value=18, allowed_values=180/np.array((180,90,60,45,30,20,15,10,5,3,2,1)))
        self.eq_nx = MidiControl(self, "eq_nx", 82, value=4, allowed_values=np.array((4,8,16,32)))
        self.eq_bins = MidiControl(self, "eq_bins", 83, value=4, allowed_values=np.power(2,np.arange(1,15)))


    def process(self, warped_frame):
        self.intermediates = []
        h,w = warped_frame.shape[0:2]
        y,u,v = split_yuv(warped_frame)
        lower_margin = 8
        #y_eq,u_eq,v_eq = equalize_adapthist_channel(y,u,v, )
        #y_eq,u_eq,v_eq = equalize_adapthist_channel(y,u,v, clip_limit=self.eq_limit.value, nbins=self.eq_bins.value, kernel_size=(int(self.eq_ny.value),int(self.eq_nx.value)))
        y_eq,u_eq,v_eq = equalize_channel(y,u,v)

        kernel3 = np.ones((3,3),np.uint8)
        kernel5 = np.ones((5,5),np.uint8)

        #----------- yellow -------------#
        y_y = binarize_img(y_eq, self.y_y_min.value, 255)
        y_y = dilate(y_y, 3)
        y_u = binarize_img(u_eq, self.y_u_min.value, 255)
        y_u = dilate(y_u, 3)
        y_v = binarize_img(v_eq, 0, self.y_v_max.value)
        y_v = dilate(y_v, 3)
        yellow = AND(y_y,y_u,y_v)
        yellow[h-lower_margin:h,:] *= 0
        mag_v_eq = mag_grad(v, self.mag_v_min.value, self.mag_v_max.value, ksize=self.mag_v_ksize.value)
        y_mag_v = AND(yellow,mag_v_eq)

        #------------ white -------------#
        w_y = binarize_img(y_eq, self.w_y_min.value, 255)
        w_y = dilate(w_y, 3)
        u_minus_v = abs_diff_channels(u,v)
        w_uv = binarize_img(u_minus_v, 0, self.w_uv_max.value)
        w_uv = dilate(w_uv, 3)

        white = AND(w_y,w_uv)
        white[h-lower_margin:h,:] *= 0
        #white = cv2.dilate(white,kernel5,iterations=1)

        mag_y_eq = mag_grad(y, self.mag_y_min.value, self.mag_y_max.value, ksize=self.mag_y_ksize.value)
        #mag_u_eq = mag_grad(u_eq, self.mag_u_min.value, self.mag_u_max.value, ksize=self.mag_u_ksize.value)

        w_mag_y = AND(white,mag_y_eq)

        th = self.dir_y_dir.value
        dth = self.dir_y_delta.value
        dir_y_eq = dir_grad(y_eq, th - 0.5 * dth, th + 0.5 * dth, ksize=self.dir_y_ksize.value)
        dir_y_eq = NOT(dir_y_eq)
        w_dir_y = AND(white,dir_y_eq)


        for ch in "y,u,v".split(","):
            self.add_intermediate_channel(eval(ch),ch)

        for ch in "y_y,y_u,y_v,yellow,y_mag_v".split(","):
            self.add_intermediate_mask(eval(ch),ch)


        for ch in "y_eq,u_eq,v_eq,u_minus_v".split(","):
            self.add_intermediate_channel(eval(ch),ch)

        for ch in "w_y,w_uv,white,w_mag_y,w_dir_y".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        #self.add_empty_intermediate()

        for ch in "mag_y_eq,dir_y_eq,mag_v_eq".split(","):
            self.add_intermediate_mask(eval(ch),ch)

        return OR(w_mag_y, y_mag_v)


class HSVPipeline(FilterPipeline):
    def __init__(self):
        super().__init__()


    def process(self, warped_frame):
        self.intermediates = []
        h,s,v = split_hsv(warped_frame)
        self.add_intermediate_channel(h,"S")
        self.add_intermediate_channel(h,"V")

        s_eq,v_eq = equalize_adapthist_channel(s,v, clip_limit=0.02, nbins=4096, kernel_size=(15,4))
        self.add_intermediate_channel(s_eq,"EQ(S)")
        self.add_intermediate_channel(v_eq,"EQ(V)")


        mag_v = mag_grad(v_eq, 32, 255, ksize=3)
        mag_s = mag_grad(s_eq, 32, 255, ksize=3)
        mag = AND(mag_v, mag_s)
        self.add_intermediate_mask(mag_v,"mag_v")
        self.add_intermediate_mask(mag_s,"mag_s")
        self.add_intermediate_mask(mag,"mag")

        s_mask = NOT(binarize_img(s_eq, 32, 175))
        v_mask = binarize_img(v_eq, 160, 255)
        mag_and_s_mask = AND(mag_v, binarize_img(v_eq, 128, 255),s_mask)
        self.add_intermediate_mask(s_mask,"s_mask")
        self.add_intermediate_mask(v_mask,"v_mask")

        center_angle = 0.5
        delta_angle = 0.5
        alpha1 = center_angle - 0.5*delta_angle
        alpha2 = alpha1 + delta_angle
        dir = NOT(dir_grad(v_eq, alpha1, alpha2, ksize=3))
        mag_and_dir = AND(mag, dir)
        self.add_intermediate_mask(dir,"dir_v")
        self.add_intermediate_mask(mag_and_dir,"dir_v & mag")

        return AND(mag_and_s_mask, v_mask)
