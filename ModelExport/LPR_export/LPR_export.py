from numpy.lib.function_base import copy
import onnx
from onnx import shape_inference
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from torch._C import dtype
from onnx_model import ONNXModel
import os
import numpy as np
from onnxsim import simplify
import cv2
from datetime import datetime

from test_model import LPRNet


CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '港', '学', '使', '警', '澳', '军', '空', '海', '领',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

class MaxPool1_3d(nn.Module):
    def __init__(self, kernel_size, stride, padding=(0,0,0)):
        super(MaxPool1_3d, self).__init__()
        assert (kernel_size[0] == 1), "kernal_size in 0 dim must be 1"
        assert (padding[0] == 0), "pandding in 0 dim must be 0"
        self.stride = stride
        self.max_pool_2d = nn.MaxPool2d(kernel_size[1:], self.stride[1:], padding[1:])

    def forward(self, x):
        # y = x[..., 0::self.stride[0], :,:]
        # print("===========", type(x.size()[1]))
        y = x.index_select(1, torch.arange(0, x.size()[1], self.stride[0], dtype=torch.int64))
        return self.max_pool_2d(y)


def PostProcess(prebs):
    # prebs = out[0]
    print(prebs.shape)
    preb_labels = list()
    preb_confs = list()
    for i in range(prebs.shape[0]):
        preb = prebs[i, :, :]
        preb_label = list()
        preb_conf = list()

        for j in range(preb.shape[1]):
            max_loc = np.argmax(preb[:, j], axis=0)
            preb_label.append(max_loc)
            preb_conf.append(preb[max_loc, j])

        no_repeat_blank_label = list()
        no_repeat_blank_conf = list()

        pre_c = preb_label[0]
        if pre_c != len(CHARS) - 1:
            no_repeat_blank_label.append(pre_c)
            no_repeat_blank_conf.append(preb_conf[0])

        for k in range(len(preb_label)): # dropout repeate label and blank label
            c = preb_label[k]
            if (pre_c == c) or (c == len(CHARS) - 1):
                if c == len(CHARS) - 1:
                    pre_c = c
                continue
            no_repeat_blank_label.append(c)
            no_repeat_blank_conf.append(preb_conf[k])
            pre_c = c
        preb_labels.append(no_repeat_blank_label)
        preb_confs.append(no_repeat_blank_conf)

        lpr_name = ''
        for lab in preb_labels[0]:
            lpr_name += CHARS[lab]

        conf = 1.0
        for f in preb_confs[0]:
            print(f,end=", ")
            conf *= f
        print("")
        print(conf, lpr_name)


def CheckOnnx(input_data, save_path):
    # Checks
    onnx_model = onnx.load(save_path)  # load onnx model
    onnx.checker.check_model(onnx_model)  # check onnx model
    # print("Check onnx done")

    onnx_runner = ONNXModel(save_path)
    out = onnx_runner.forward(input_data.numpy())
    return out[0]



def load_static_weight(model, weight):
    state_dict = torch.load(weight)
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:6] == 'module':
            name = k[7:]
        else:
            name = k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    return model.float().eval()

def update_model(model):
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, nn.MaxPool3d):  # assign export-friendly MaxPool3d
            print(k, "->", m)
            idx = str(k).split(".")[1]
            model.backbone[int(idx)] = MaxPool1_3d(m.kernel_size, m.stride)
            
    return model


def export_onnx(model, img, save_path, opset_version=11):
    try:
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        torch.onnx.export(model, img, save_path, verbose=False, opset_version=opset_version, input_names=['images'],
                          do_constant_folding=True,
                          output_names=["output",])

        # Checks
        onnx_model = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        date_time = datetime.now().strftime("%Y%m%d")
        onnx_sim_path = os.path.splitext(save_path)[0] + "-" + date_time + "-sim.onnx"
        model_simp = onnx.shape_inference.infer_shapes(model_simp)
        onnx.save(model_simp, onnx_sim_path)
        print('ONNX export success, saved as %s' % onnx_sim_path)
        return onnx_sim_path
    except Exception as e:
        print('ONNX export failure: %s' % e)
        return None


def ImagePreprocess(img_src):
    img = cv2.resize(img_src, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
    org_img = copy(img)
    # img = cv2.medianBlur(img, 3) 
    img = cv2.GaussianBlur(img, (5, 5), 75)  
    # cv2.imshow("blurImage", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img_mask = cv2.dilate(img, kernel, iterations=1)
    img_mask = cv2.erode(img_mask, kernel, iterations=1)
    img_mask = cv2.adaptiveThreshold(cv2.medianBlur(img_mask, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    img = cv2.bitwise_and(org_img, org_img, mask=img_mask)
    # cv2.imshow("Image", img)
    # cv2.imshow("orgImage", org_img)
    # cv2.waitKey(0)
    return img


def Softmax(in_data):
    # in_data = torch.Tensor(in_data).float()
    in_data = torch.from_numpy(in_data)
    out_data = F.softmax(in_data, 1)
    return out_data.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./LPRNet__iteration_45000.pth', help='weights path')  # from yolov5/models/
    opt = parser.parse_args()
    print(opt)

    #  input data
    img = cv2.imread("./align_plate_w.jpg")
    # img = ImagePreprocess(img)
    org_img = copy(img)
    # exit(0)
    
    img = cv2.resize(img, (94,24))
    img = img.astype('float32')
    img -= 128
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    # print("image: \n", img)

    img = torch.from_numpy(img)
    if img.dim() == 3:
        img = torch.unsqueeze(img, 0)
    print(img.shape)

    # load model as eval
    model = LPRNet(lpr_max_len=8, phase=False, class_num=77, dropout_rate=0.5)
    model = load_static_weight(model, opt.weights)
    
    print("================= LPR Model =================")
    print("===================== org py")
    y = model(img)  # dry run
    prebs = y.detach().numpy()
    # print(prebs)
    preb_confs = Softmax(prebs)
    # print(preb_confs)
    PostProcess(preb_confs)

    print("===================== update py")
    # update model use custom MaxPool3d instand of nn.MaxPool3d
    model = update_model(model)
    y = model(img)  # dry run
    prebs = y.detach().numpy()
    # print(prebs)
    preb_confs = Softmax(prebs)
    print(preb_confs)
    PostProcess(preb_confs)

    # export onnx
    onnx_export_file = os.path.splitext(opt.weights)[0] + "_" + str(94) + "_" + str(24) + ".onnx" 
    onnx_sim_path = export_onnx(model, img, onnx_export_file, opset_version=11)

    # # infer use onnxruntime
    # print("===================== onnx")
    # onnx_sim_path = "./LPRNet__iteration_45000-sim.onnx"
    # prebs = CheckOnnx(img, onnx_sim_path)
    # # np.savetxt("prebs.txt", prebs[0,:,:])
    # # print(prebs)
    # preb_confs = Softmax(prebs)
    # # print(preb_confs)
    # # np.savetxt("preb_confs.txt", preb_confs[0,:,:])
    # PostProcess(preb_confs)
