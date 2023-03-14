import cv2
import numpy as np
from facenet_pytorch import MTCNN
from angle_estimation import estimate_angles
import imutils


class Face:
    def init(self):
        self.name = None
        self.bounding_box = None
        self.prob = None
        self.image = None
        self.container_image = None
        self.keypoints = None
        self.angles = None
        self.blur = None
        self.area_ratio = None
        self.position = None

    def is_clear(self):
        if self.blur is None:
            raise Exception("Blurriness not presented")
        if self.blur < 200 or self.angles is None:
            return False
        yaw, pitch, roll = \
            abs(self.angles["yaw"]), abs(self.angles["pitch"]), abs(self.angles["roll"])

        return yaw < 75 and pitch < 75 and roll < 75 and yaw + pitch + roll < 180


class Detector:

    def __init__(self, face_crop_size=160, face_crop_margin=32, face_min_score=0.5, face_min_height=32, device=None):
        self.device = device
        self.face_detector = MTCNN(keep_all=True, post_process=False,
                                   image_size=face_crop_size,
                                   margin=face_crop_margin,
                                   min_face_size=face_min_height,
                                   device=self.device)

        self.face_crop_size = face_crop_size
        self.face_crop_margin = face_crop_margin
        self.face_min_score = face_min_score
        self.face_min_height = face_min_height

    def detect(self, original_image,
               mode="rgb",
               face_crop_margin=-1,
               scale_factor=1,
               crop_faces=True,
               angle_estimation=False):
        faces = []
        if scale_factor > 1:
            scale_factor = 1
        if face_crop_margin == -1:
            face_crop_margin = self.face_crop_margin

        if mode == "rgb":
            rgb_image = original_image
        else:
            rgb_image = original_image[..., ::-1].copy()

        image_height, image_width = rgb_image.shape[:2]
        image_area = image_width * image_height

        bounding_boxes, probs, landmarks = self.face_detector.detect(rgb_image, landmarks=True)

        if bounding_boxes is None or len(bounding_boxes) == 0:
            return faces

        for bb_index, bb in enumerate(bounding_boxes):
            score = probs[bb_index]
            points = landmarks[bb_index]
            keypoints = dict()
            keypoints["left_eye"], keypoints["right_eye"], keypoints["nose"], keypoints["mouth_left"], keypoints[
                "mouth_right"] = points
            x1, y1, x2, y2 = bb

            if score > self.face_min_score and (y2 - y1) > self.face_min_height * scale_factor:
                face = Face()
                face.container_image = original_image
                face.bounding_box = np.zeros(4, dtype=np.int32)
                face.keypoints = keypoints
                face_area = (y2 - y1) * (x2 - x1)
                face_center_x = (x1 + x2) / 2.0
                face_center_y = (y1 + y2) / 2.0
                face.area_ratio = round(face_area / float(image_area) * 100, 2)
                position_x = round(face_center_x / float(image_width) * 100, 2)
                position_y = round(face_center_y / float(image_height) * 100, 2)
                face.position = {'x': position_x, 'y': position_y}

                if angle_estimation:
                    estimate_angles(original_image, face)
                    cropped_bb = np.zeros(4, dtype=np.int32)
                    cropped_bb[0] = np.ceil(np.maximum(x1 - face_crop_margin / 2, 0))
                    cropped_bb[1] = np.ceil(np.maximum(y1 - face_crop_margin / 2, 0))
                    cropped_bb[2] = np.ceil(np.maximum(x2 + face_crop_margin / 2, 0))
                    cropped_bb[3] = np.ceil(np.maximum(y2 + face_crop_margin / 2, 0))
                    face.bounding_box[0] = np.ceil(cropped_bb[0] / scale_factor)
                    face.bounding_box[1] = np.ceil(cropped_bb[1] / scale_factor)
                    face.bounding_box[2] = np.ceil(cropped_bb[2] / scale_factor)
                    face.bounding_box[3] = np.ceil(cropped_bb[3] / scale_factor)
                    face.prob = score
                    cropped = original_image[cropped_bb[1]:cropped_bb[3], cropped_bb[0]:cropped_bb[2], :]
                    face.blur = self.__measure_blur(cropped)
                    # face.blur = self.__detect_blur_fft(cropped)
                    if crop_faces:
                        # face.image = misc.imresize(cropped, (self.face_crop_size, self.face_crop_size), interp='bilinear')

                        # order = 1 is a bilinear interpolation
                        # face.image = skimage.transform.resize(cropped, (self.face_crop_size, self.face_crop_size),
                        #                                       order=1, preserve_range=True).astype(np.uint8)
                        face.image = cv2.resize(cropped, (self.face_crop_size, self.face_crop_size),
                                                interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    faces.append(face)

        return faces

    def __measure_blur(self, cropped):
        grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        return np.percentile(cv2.Laplacian(grey, cv2.CV_64F), 99)

    def __detect_blur_fft(self, image, size=60):
        image = imutils.resize(image, width=200)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # grab the dimensions of the image and use the dimensions to
        # derive the center (x, y)-coordinates
        (h, w) = image.shape
        (cX, cY) = (int(w / 2.0), int(h / 2.0))
        # compute the FFT to find the frequency transform, then shift
        # the zero frequency component (i.e., DC component located at
        # the top-left corner) to the center where it will be more
        # easy to analyze
        fft = np.fft.fft2(image)
        fftShift = np.fft.fftshift(fft)
        # zero-out the center of the FFT shift (i.e., remove low
        # frequencies), apply the inverse shift such that the DC
        # component once again becomes the top-left, and then apply
        # the inverse FFT
        fftShift[cY - size:cY + size, cX - size:cX + size] = 0
        fftShift = np.fft.ifftshift(fftShift)
        recon = np.fft.ifft2(fftShift)
        # compute the magnitude spectrum of the reconstructed image,
        # then compute the mean of the magnitude values
        magnitude = 20 * np.log(np.abs(recon))
        mean = np.mean(magnitude)
        # the image will be considered "blurry" if the mean value of the
        # magnitudes is less than the threshold value
        return mean
