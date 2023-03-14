# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:58:14 2018

@author: jerry
"""

import cv2
import math
import numpy as np

face_part_2_model_point = {"nose": (0.0, 0.0, 0.0),
                           "chin": (0.0, -330.0, -65.0),
                           # "left_eye": (-165.0, 170.0, -135.0),
                           # "right_eye": (165.0, 170.0, -135.0),
                           "left_eye": (-150.0, 170.0, -125.0),
                           "right_eye": (150.0, 170.0, -125.0),
                           "mouth_left": (-145.0, -150.0, -125.0),
                           "mouth_right": (145.0, -150.0, -125.0)}


def estimate_angles(frame, face):
    size = frame.shape  # (height, width, color_channel)
    keypoints = face.keypoints
    solver = cv2.SOLVEPNP_SQPNP
    face_parts = ("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")
    angles = compute_angles(keypoints, face_parts, size, solver=solver)
    face.angles = dict(zip(["pitch", "yaw", "roll"], angles))


def compute_angles(keypoints, face_parts, size, solver=cv2.SOLVEPNP_AP3P):
    image_points, model_points = get_points(keypoints, face_parts)
    # Camera internals
    center = (size[1] / 2, size[0] / 2)
    focal_length = center[0] / np.tan(60 / 2 * np.pi / 180)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=solver)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = [math.radians(_) for _ in euler_angles]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return [pitch, yaw, roll]


def get_points(keypoints, face_parts=("nose", "left_eye", "right_eye", "mouth_left", "mouth_right")):
    image_points = np.array([keypoints[part] for part in face_parts], dtype="double")
    model_points = np.array([face_part_2_model_point[part] for part in face_parts])
    return image_points, model_points