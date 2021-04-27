import cv2
import numpy as np
import os
import torch

class LPR():
    def __init__(self, folder):
        refine_net_path = [os.path.join(folder,"./refinenet.prototxt"),os.path.join(folder,"./refinenet.caffemodel")]
        self.refine_net = cv2.dnn.readNetFromCaffe(*refine_net_path)

    def to_refine(self,image, pts, scale=3.0):
        """
        refine the image by input points.
        :param image_rgb: input image
        :param pts: points
        """
        x1, y1, x2, y2, x3, y3, x4, y4 = pts.ravel()
        cx, cy = int(128 // 2), int(48 // 2)
        cw = 64
        ch = 24
        tx1 = cx - cw // 2
        ty1 = cy - ch // 2
        tx2 = cx + cw // 2
        ty2 = cy - ch // 2
        tx3 = cx + cw // 2
        ty3 = cy + ch // 2
        tx4 = cx - cw // 2
        ty4 = cy + ch // 2
        target_pts = np.array([[tx1, ty1], [tx2, ty2], [tx3, ty3], [tx4, ty4]]).astype(np.float32) * scale
        org_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(np.float32)
        mat_ = cv2.estimateRigidTransform(org_pts, target_pts, True)
        dsize = (int(120 * scale), int(48 * scale))
        warped = cv2.warpAffine(image, mat_, dsize)
        return warped

    def affine_crop(self, image, pts, size):
        """
        crop a image by affine transform.
        :param image_rgb: input image
        :param pts: points
        """
        w = size[0]
        h = size[1]
        x1, y1, x2, y2, x3, y3, x4, y4 = pts.ravel()
        target_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]]).astype(np.float32)
        org_pts = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]]).astype(np.float32)
        mat = cv2.getPerspectiveTransform(org_pts, target_pts)
        dsize = (w, h)
        warped = cv2.warpPerspective(image, mat, dsize)
        return warped

    def finetune(self,image_, size=(160,40), stage=2):
        """
        cascade fine tune a image by regress four corner of plate.
        :param image_rgb: input image
        :param stages: cascade stage
        """

        tof = image_.copy()
        image = cv2.resize(tof, (120, 48))
        blob = cv2.dnn.blobFromImage(image, size=(120, 48), swapRB=False, mean=(127.5, 127.5, 127.5), scalefactor=0.0078125, crop=False)
        self.refine_net.setInput(blob)
        h, w, c = image_.shape
        pts = (self.refine_net.forward("conv6-3").reshape(4, 2) * np.array([w, h])).astype(np.int)
        g = self.to_refine(image_, pts)
        blob = cv2.dnn.blobFromImage(g, size=(120, 48), swapRB=False, mean=(127.5, 127.5, 127.5), scalefactor=0.0078125, crop=False)
        self.refine_net.setInput(blob)
        h, w, c = g.shape
        pts = (self.refine_net.forward("conv6-3").reshape(4, 2) * np.array([w, h])).astype(np.int)
        cropped = self.affine_crop(g, pts, size)
        return cropped

lpr = LPR(os.path.join(os.path.split(os.path.realpath(__file__))[0], "."))    
img = cv2.imread('皖A4Y389.jpg')
align_image = lpr.finetune(img)
cv2.imshow('img',img)
cv2.waitKey(0)
    
