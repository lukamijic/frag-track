import cv2
import integral_images as ii
from patches import Patches
import numpy as np


class FragTracker:

    DEFAULT_VIDEO_PATH = "videos/times_square2.mp4"
    DEFAULT_SPLIT = (10, 10)
    DEFAULT_RADIUS = 20

    def __init__(self, video_path=DEFAULT_VIDEO_PATH, split=DEFAULT_SPLIT, radius=DEFAULT_RADIUS):
        self.video_path = video_path
        self.video = cv2.VideoCapture(self.video_path)
        self.split = split
        self.radius = radius

        ok, frame = self.video.read()
        self.frame_width = len(frame[0])
        self.frame_height = len(frame)

        if not ok:
            raise RuntimeError("First frame not available")

        self.bound_box = cv2.selectROI(frame, False)
        print(self.bound_box)
        self.template_width = self.bound_box[2]
        self.template_height = self.bound_box[3]
        self.t_half_width = int(self.template_width / 2)
        self.t_half_height = int(self.template_height / 2)
        end_x, end_y = self.bound_box[0] + self.template_width, self.bound_box[1] + self.template_height
        template = frame[self.bound_box[1]:end_y, self.bound_box[0]:end_x]
        template = cv2.cvtColor(template, cv2.COLOR_BGR2HSV)
        integral_bins = ii.hue_integral_bins(template)
        self.template_patches = Patches(integral_bins, 0, 0, self.template_width, self.template_height
                                        , split= self.split)

    def execute(self, step=1):
        center_x = int(self.bound_box[0] + self.t_half_width)
        center_y = int(self.bound_box[1] + self.t_half_height)
        create_video = cv2.VideoWriter('results/times_square22.mp4', cv2.VideoWriter_fourcc('M','J','P','G')
                                       , 30, (self.frame_width, self.frame_height))
        while True:
            ok, frame = self.video.read()
            if not ok:
                break
            integral_bins, new_center = self.calculate_needed_integral_bins(frame, center_x, center_y)
            c_x = new_center[0]
            c_y = new_center[1]

            min_dist = None
            for j in range(-self.radius, self.radius + 1, step):
                for i in range(-self.radius, self.radius + 1, step):
                    x, y = center_x + i, center_y + j
                    if not self.check_is_rectangle_in_bounds(x - self.t_half_width, y - self.t_half_height):
                        continue
                    start_x = c_x + i - self.t_half_width
                    start_y = c_y + j - self.t_half_height
                    new_bb = (x - self.t_half_width, y - self.t_half_height,
                              self.template_width, self.template_height)
                    new_patches = Patches(integral_bins, start_x, start_y, self.template_width, self.template_height,
                                          split=self.split)
                    d = self.template_patches.distance(new_patches)
                    if min_dist is None:
                        min_dist = (d, new_bb)
                    if min_dist[0] > d:
                        min_dist = (d, new_bb)
            new_bb = min_dist[1]
            left_top = (new_bb[0], new_bb[1])
            right_bot = (new_bb[0] + new_bb[2], new_bb[1] + new_bb[3])
            cv2.rectangle(frame, left_top, right_bot, (255, 0, 0), 2, 1)
            cv2.imshow("Tracking", frame)
            create_video.write(frame)

            center_x = int(new_bb[0] + new_bb[2] / 2)
            center_y = int(new_bb[1] + new_bb[3] / 2)

            k = cv2.waitKey(1) & 0xff
            if k == 27: break
        create_video.release()
        cv2.destroyAllWindows()

    def calculate_needed_integral_bins(self, frame, center_x, center_y):
        start_x = center_x - self.radius - self.t_half_width
        start_y = center_y - self.radius - self.t_half_height
        end_x = center_x + self.radius + self.t_half_width + 1
        end_y = center_y + self.radius + self.t_half_height + 1

        s_x, s_y, e_x, e_y = self.calibrate_bounds(start_x, start_y, end_x, end_y)
        frame = frame[s_y:e_y, s_x:e_x]

        hues = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hues = cv2.split(hues)[0]
        y_pad_top, y_pad_bot = (0, 0)
        x_pad_left, x_pad_right = (0, 0)
        if start_y < 0:
            y_pad_top = -1 * start_y
        if start_y > self.frame_height - 1:
            y_pad_bot = end_y - self.frame_height
        if start_x < 0:
            x_pad_left = -1 * start_x
        if end_x > self.frame_width - 1:
            x_pad_right = end_x - self.frame_width

        y_padding = (y_pad_top, y_pad_bot)
        x_padding = (x_pad_left, x_pad_right)
        hues = np.pad(hues, [y_padding, x_padding], 'constant')
        shape = hues.shape
        new_center = int(shape[1] / 2), int(shape[0] / 2)
        return ii.hue_integral_bins(hues), new_center

    def calibrate_bounds(self, start_x, start_y, end_x, end_y):
        if start_x < 0:
            start_x = 0
        elif start_x >= self.frame_width:
            start_x = self.frame_width - 1

        if end_x < 0:
            end_x = 0
        elif end_x >= self.frame_width:
            end_x = self.frame_width - 1

        if start_y < 0:
            start_y = 0
        elif start_y >= self.frame_height:
            start_y = self.frame_height - 1

        if end_y < 0:
            end_y = 0
        elif end_y >= self.frame_height:
            end_y = self.frame_height - 1
        return start_x, start_y, end_x, end_y

    def is_in_bounds(self, x, y):
        if x < 0 or x >= self.frame_width or y < 0 or y > self.frame_height:
            return False
        return True

    def check_is_rectangle_in_bounds(self, i, j):
        if not self.is_in_bounds(i, j):
            return False
        if not self.is_in_bounds(i + self.template_width, j + self.template_height):
            return False
        return True
