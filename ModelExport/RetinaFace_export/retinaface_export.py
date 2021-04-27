from __future__ import print_function
import os
import argparse
from numpy.core.fromnumeric import shape
import torch
from torch import nn
import torch.nn.functional as F
from torch._C import dtype
import torch.backends.cudnn as cudnn
import numpy as np
from itertools import chain
from datetime import datetime
# from data import cfg_mnet, cfg_re50
# from layers.functions.prior_box import PriorBox
# from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
# from utils.box_utils import decode, decode_landm
import time

import onnx
from onnx import numpy_helper, mapping


cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': False,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': False,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': True,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}


def GetConstantInfoByName(constant_name, onnx_model):
    contant_size = len(onnx_model.graph.initializer)
    for j in range(contant_size):
        constant = onnx_model.graph.initializer[j]

        if constant.name == constant_name:
            print(constant.name, ": ", numpy_helper.to_array(constant))
            shape_dim = constant.dims
            shape_type = constant.data_type
            np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE[shape_type]
            shape_data = np.frombuffer(constant.raw_data, dtype=np_dtype).reshape(shape_dim)

            return shape_data, np_dtype


def RemoveConstantByName(constant_name, onnx_model):
    contant_size = len(onnx_model.graph.initializer)
    rm_cnt = 0
    for j in range(contant_size):
        j -= rm_cnt
        constant = onnx_model.graph.initializer[j]
        if constant.name in constant_name:
            print(constant.name, ": ", numpy_helper.to_array(constant))
            onnx_model.graph.initializer.pop(j)
            rm_cnt += 1



def AddNewConstant(new_constant_name, shape_data_new, onnx_model):
    shape_contant_new = numpy_helper.from_array(shape_data_new, new_constant_name)
    onnx_model.graph.initializer.append(shape_contant_new)


def ModifyDynamicBatch(onnx_model):
    # 修改输入为动态batch
    model_input = onnx_model.graph.input[0]
    model_input.type.tensor_type.shape.dim[0].dim_value = -1
    # 修改Reshape层操作为固定输入尺寸的动态batch参数
    reshape_1_name = ["Reshape_115", "Reshape_145"]
    reshape_out_8_name = ["Reshape_224", "Reshape_199", "Reshape_249"]
    reshape_out_16_name = ["Reshape_232", "Reshape_207", "Reshape_257"]
    reshape_out_32_name = ["Reshape_240", "Reshape_215", "Reshape_265"]

    shape_8_mid_size = 12800
    shape_16_mid_size = 3200
    shape_32_mid_size = 800

    shape_1_name = []
    shape_out_8_name = []
    shape_out_16_name = []
    shape_out_32_name = []

    pop_contant_names = []

    node_cnt = len(onnx_model.graph.node)
    for i in range(node_cnt):
        node = onnx_model.graph.node[i]
        if node.name in reshape_1_name:
            shape_name = node.input[1]
            pop_contant_names.append(shape_name)
            shape_data, np_dtype = GetConstantInfoByName(shape_name, onnx_model)
            new_constant_name = node.name + "_" + shape_name
            shape_data_new = np.append(np.array([-1,], dtype=np_dtype), shape_data[1:])
            shape_contant_new = numpy_helper.from_array(shape_data_new, new_constant_name)
            onnx_model.graph.initializer.append(shape_contant_new)
            node.input[1] = new_constant_name
            shape_1_name.append(new_constant_name)

        elif node.name in reshape_out_8_name:
            shape_name = node.input[1]
            pop_contant_names.append(shape_name)
            shape_data, np_dtype = GetConstantInfoByName(shape_name, onnx_model)
            new_constant_name = node.name + "_" + shape_name
            shape_data_new = np.array([-1, shape_8_mid_size, shape_data[-1]], dtype=np_dtype)
            shape_contant_new = numpy_helper.from_array(shape_data_new, new_constant_name)
            onnx_model.graph.initializer.append(shape_contant_new)
            node.input[1] = new_constant_name
            shape_out_8_name.append(new_constant_name)
        elif node.name in reshape_out_16_name:
            shape_name = node.input[1]
            pop_contant_names.append(shape_name)
            shape_data, np_dtype = GetConstantInfoByName(shape_name, onnx_model)
            new_constant_name = node.name + "_" + shape_name
            shape_data_new = np.array([-1, shape_16_mid_size, shape_data[-1]], dtype=np_dtype)
            shape_contant_new = numpy_helper.from_array(shape_data_new, new_constant_name)
            onnx_model.graph.initializer.append(shape_contant_new)
            node.input[1] = new_constant_name
            shape_out_16_name.append(new_constant_name)
        elif node.name in reshape_out_32_name:
            shape_name = node.input[1]
            pop_contant_names.append(shape_name)
            shape_data, np_dtype = GetConstantInfoByName(shape_name, onnx_model)
            new_constant_name = node.name + "_" + shape_name
            shape_data_new = np.array([-1, shape_32_mid_size, shape_data[-1]], dtype=np_dtype)
            shape_contant_new = numpy_helper.from_array(shape_data_new, new_constant_name)
            onnx_model.graph.initializer.append(shape_contant_new)
            node.input[1] = new_constant_name
            shape_out_32_name.append(new_constant_name)

    RemoveConstantByName(pop_contant_names, onnx_model)

    print("=====================================")
    modify_reshape_name = chain(shape_1_name, shape_out_8_name, shape_out_16_name, shape_out_32_name)
    modify_reshape_name = list(modify_reshape_name)

    for k in onnx_model.graph.initializer:
        if str(k.name) in modify_reshape_name:
            print(numpy_helper.to_array(k))
   

def export_onnx(model, img, save_path, opset_version=11):
    try:
        import onnx
        from onnxsim import simplify
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        torch.onnx.export(model, img, save_path, verbose=False, opset_version=opset_version, input_names=['images'],
                          do_constant_folding=True,
                          output_names=['feature_map1', 'feature_map2', 'feature_map3'])

        # Checks
        onnx_model = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        print('ONNX export success, saved as %s' % save_path)
        
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        date_time = datetime.now().strftime("%Y%m%d")
        onnx_sim_path = os.path.splitext(save_path)[0] + "-" + date_time + "-sim.onnx"
        # ModifyDynamicBatch(model_simp)
        # onnx_sim_path = os.path.splitext(save_path)[0] + "-dynamic.onnx"
        # return
        model_simp = onnx.shape_inference.infer_shapes(model_simp)
        onnx.save(model_simp, onnx_sim_path)
        print('ONNX simplify export success, saved as %s' % onnx_sim_path)
    except Exception as e:
        print('ONNX export failure: %s' % e)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True

def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}

def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./mobilenet0.25_Final.pth', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)

    torch.set_grad_enabled(False)
    cfg = None
    cfg = cfg_mnet
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, opt.weights, True)
    net.eval()
    print('Finished loading model!')
    device = torch.device("cpu")
    net = net.to(device).float()

    img = torch.zeros([1, 3, *opt.img_size], dtype=torch.float32)

    y = net(img)

    onnx_export_file = os.path.splitext(opt.weights)[0] + "_" + str(opt.img_size[0]) + "_" + str(opt.img_size[1]) + ".onnx" 
    export_onnx(net, img, onnx_export_file)
    

