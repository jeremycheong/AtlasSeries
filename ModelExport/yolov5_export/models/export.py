"""Exports a YOLOv5 *.pt model to ONNX and TorchScript formats

Usage:
    $ export PYTHONPATH="$PWD" && python models/export.py --weights ./weights/yolov5s.pt --img 640 --batch 1
"""

import argparse
import sys
import time
import os
from datetime import datetime
import copy

from onnxsim import simplify

sys.path.append('./')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import onnx

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from models.common import NearestUpsample
from utils.general import set_logging, check_img_size
from utils.torch_utils import time_synchronized, fuse_conv_and_bn


def RemoveNode(graph, node_list):
    max_idx = len(graph.node)
    rm_cnt = 0
    for i in range(len(graph.node)):
        if i - rm_cnt < max_idx:
            node = graph.node[i - rm_cnt]
            if node.name in node_list:
                print("remove {} total {}".format(node.name, len(graph.node)))
                graph.node.remove(node)
                max_idx -= 1
                rm_cnt += 1

def ReplaceScales(ori_list, scales_name):
    n_list = []
    for i, x in enumerate(ori_list):
        if i < 2:
            n_list.append(x)
        if i == 2:
            n_list.append(scales_name)
    return n_list

def StaticResize(model):
    # 替换Resize节点
    for i in range(len(model.graph.node)):
        Node = model.graph.node[i]
        if Node.op_type == "Resize":
            print("Resize", i, Node.input, Node.output)
            model.graph.initializer.append(
                onnx.helper.make_tensor('scales{}'.format(i), onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
            )
            newnode = onnx.helper.make_node(
                'Resize',
                name=Node.name,
                inputs=ReplaceScales(Node.input, 'scales{}'.format(i)),
                outputs=Node.output,
                coordinate_transformation_mode='asymmetric',
                cubic_coeff_a=-0.75,
                mode='nearest',
                nearest_mode='floor'
            )
            model.graph.node.remove(model.graph.node[i])
            model.graph.node.insert(i, newnode)
            print("replace {} index {}".format(Node.name, i))


def modify_yolov5_onnx(onnx_model_path, image_h, image_w):
    model = onnx.load(onnx_model_path)
    # prob_info = onnx.helper.make_tensor_value_info('images', onnx.TensorProto.FLOAT, [1, 12, image_h // 2, image_w // 2])
    # model.graph.input.remove(model.graph.input[0])
    # model.graph.input.insert(0, prob_info)
    # model.graph.node[41].input[0] = 'images'

    # node_list = ["Concat_40"]

    # slice_node = ["Slice_4", "Slice_14", "Slice_24", "Slice_34", "Slice_9", "Slice_19", "Slice_29", "Slice_39", ]
    # node_list.extend(slice_node)

    # RemoveNode(model.graph, node_list)

    # # StaticResize(model)

    # constant_nodes = []
    # for i in range(39):
    #     constant_nodes.append("Constant_" + str(i))
    
    # RemoveNode(model.graph, constant_nodes)

    onnx.checker.check_model(model)
    save_file = os.path.splitext(onnx_model_path)[0] + "_modify.onnx"

    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    date_time = datetime.now().strftime("%Y%m%d")
    onnx_sim_path = os.path.splitext(save_file)[0] + "-" + date_time + ".onnx"
    onnx.save(model_simp, onnx_sim_path)

def relace_focus(model):
    focus2_model = models.common.Focus2(3, 64)
    focus_conv = None
    for idx, m in model.named_modules():
        if type(m) is models.common.Focus:
            focus_conv = copy.deepcopy(m.conv)

    # focus2_model.apply(lambda m: )
    for idx, m in focus2_model.named_modules():
        if type(m) is models.common.Conv and hasattr(m, "bn"):
            focus2_model.conv = copy.deepcopy(focus_conv)
            # print(" ====>>> focus2_model ", idx, m)

    for idx, m in model.named_modules():
        if type(m) is models.common.Focus:
            # setattr(model, idx, focus2_model)
            model.model[0] = copy.deepcopy(focus2_model)
            model.model[0].f = -1
            model.model[0].i = 0
            # print("====>>>repalce ",idx, m)

    print("update focus layer")

    return model


def update_model(model):
    print("===== update model")
    model = relace_focus(model)

    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, models.common.Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            # elif isinstance(m.act, nn.SiLU):
            #     m.act = SiLU()

        elif isinstance(m, nn.Upsample):
            print(k, " -> ", m)
            if k == "model.11":
                model.model[11] = NearestUpsample()  #nn.Upsample([40, 40], None, 'nearest')
                model.model[11].f = -1
                model.model[11].i = 11
            elif k == "model.15":
                model.model[15] = NearestUpsample()  #nn.Upsample([80, 80], None, 'nearest')
                model.model[15].f = -1
                model.model[15].i = 15

        # elif isinstance(m, models.yolo.Detect):
        #     m.forward = m.forward_export  # assign forward (optional)
    model.model[-1].export = True  # set Detect() layer export=True

    return model


def export_onnx(model, img, save_path, opset_version=11):
    try:
        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        torch.onnx.export(model, img, save_path, verbose=False, opset_version=opset_version, input_names=['images'],
                          do_constant_folding=True,
                          output_names=['feature_map_1', 'feature_map_2', 'feature_map_3']
                          )

        # Checks
        onnx_model = onnx.load(save_path)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        # print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % save_path)
    except Exception as e:
        print('ONNX export failure: %s' % e)


if __name__ == '__main__':
    # focus_moudle()
    # exit(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov5s.pt', help='weights path')  # from yolov5/models/
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    opt = parser.parse_args()
    opt.img_size *= 2 if len(opt.img_size) == 1 else 1  # expand
    print(opt)
    set_logging()
    t = time.time()

    # Load PyTorch model
    model = attempt_load(opt.weights, map_location=torch.device('cpu'))  # load FP32 model
    
    labels = model.names

    # Checks
    gs = int(max(model.stride))  # grid size (max stride)
    opt.img_size = [check_img_size(x, gs) for x in opt.img_size]  # verify img_size are gs-multiples

    # Input
    img = torch.zeros(opt.batch_size, 3, *opt.img_size).to(torch.device('cpu'))  # image size(1,3,320,192) iDetection

    # Update model
    model = update_model(model)
    y = model(img)  # dry run
    # print(y)

    # ONNX export
    onnx_export_file = os.path.splitext(opt.weights)[0] + "_" + str(opt.img_size[0]) + "_" + str(opt.img_size[1]) + ".onnx" 
    export_onnx(model, img, onnx_export_file)

    # modify ONNX model
    modify_yolov5_onnx(onnx_export_file, opt.img_size[0], opt.img_size[1])

    # Finish
    print('\nExport complete (%.2fs). Visualize with https://github.com/lutzroeder/netron.' % (time.time() - t))
