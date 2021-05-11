#!/usr/bin/env python3
# coding: utf-8

__author__ = 'cleardusk'

"""
The pipeline of 3DDFA prediction: given one image, predict the 3d face vertices, 68 landmarks and visualization.

[todo]
1. CPU optimization: https://pmchojnacki.wordpress.com/2018/10/07/slow-pytorch-cpu-performance
"""

import torch
import torchvision.transforms as transforms
import mobilenet_v1
import numpy as np
import cv2
import dlib
from utils.ddfa import ToTensorGjz, NormalizeGjz, str2bool
import scipy.io as sio
from utils.inference import get_suffix, parse_roi_box_from_landmark, crop_img, predict_68pts, dump_to_ply, dump_vertex, \
    draw_landmarks, predict_dense, parse_roi_box_from_bbox, get_colors, write_obj_with_colors
from utils.cv_plot import plot_pose_box
from utils.estimate_pose import parse_pose
from utils.render import get_depths_image, cget_depths_image, cpncc
from utils.paf import gen_img_paf
import argparse
import torch.backends.cudnn as cudnn
import os
import sys
import cv2 as cv

STD_SIZE = 120


def main(args):
    # 1. load pre-tained model
    #checkpoint_fp = 'models/phase1_wpdc_vdc.pth.tar'
    #arch = 'mobilenet_1'

    #checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
    #model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

    #model_dict = model.state_dict()
    # because the model is trained by multiple gpus, prefix module should be removed
    #for k in checkpoint.keys():
    #    model_dict[k.replace('module.', '')] = checkpoint[k]
    #model.load_state_dict(model_dict)
    #if args.mode == 'gpu':
    #    cudnn.benchmark = True
    #    model = model.cuda()
    #model.eval()

    # 2. load dlib model for face detection and landmark used for face cropping
    #if args.dlib_landmark:
    #    dlib_landmark_model = 'models/shape_predictor_68_face_landmarks.dat'
    #    face_regressor = dlib.shape_predictor(dlib_landmark_model)
    #if args.dlib_bbox:
    #    face_detector = dlib.get_frontal_face_detector()

    # 3. forward
    #tri = sio.loadmat('visualize/tri.mat')['tri']
    #transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])
    
    input_dir = '/media/rzats/Elements/UV_Data/Incomplete_UVs'
    img_paths = []
    for subdir, dirs, files in os.walk(input_dir):
      for file in files:
        if file.endswith('_0_2.jpg') or file.endswith('_0_5.jpg') or file.endswith('_0_8.jpg'):
          img_paths.append(file)
    img_paths = set(img_paths)
    '''
    output_dir = '/media/rzats/Elements/UV_Data/Incomplete_UVs'
    output_paths = []
    for subdir, dirs, files in os.walk(output_dir):
      for file in files:
          output_paths.append(file)
    print(img_paths[:100])
    output_paths = set(output_paths)
    imgs = []
    inputs = []
    input_paths = []
    
    # TODO: run catchup; single thread ok
    RESUME_INDEX = 0#77000
    
    floof = 0
    '''

    for i in range(100000):
        # this vertical stuff was garbage, delete later
        name = '{0:05d}'.format(i)
        print(name)
        left_name = '{}_0_8.jpg'.format(name)
        middle_name = '{}_0_5.jpg'.format(name)
        right_name = '{}_0_2.jpg'.format(name)
        if left_name in img_paths and middle_name in img_paths and right_name in img_paths:
          left = cv2.imread(input_dir + os.sep + left_name)
          center = cv2.imread(input_dir + os.sep + middle_name)
          right = cv2.imread(input_dir + os.sep + right_name)

          # Take the middle third part from the central image
          center_mid = np.array_split(center, 3, axis=1)[1]
          
          # Define mask & center point for seamlessClone
          black = np.full(center_mid.shape, 0, dtype = np.uint8)
          white = np.full(center_mid.shape, 255, dtype = np.uint8)
          mask = np.concatenate([black, white, black], axis=1)
          center_point = (left.shape[1]//2, left.shape[0]//2)
          center_masked = np.concatenate([black, center_mid, black], axis=1)

          # Overlay center middle onto left image, then right image
          im_clone1 = cv.seamlessClone(center_masked, left, mask, center_point, cv.NORMAL_CLONE)
          im_clone2 = cv.seamlessClone(center_masked, right, mask, center_point, cv.NORMAL_CLONE)

          # Merge left & right halves of the results
          clones_with_seam = np.concatenate([np.array_split(im_clone1, 2, axis=1)[0], np.array_split(im_clone2, 2, axis=1)[1]], axis=1)

          # Overlay center middle again to correct the seam in the middle
          clones_without_seam = cv.seamlessClone(center_masked, clones_with_seam, mask, center_point, cv.NORMAL_CLONE)
          seamless_path = '/media/rzats/Elements/UV_Data/Completed_UVs/{}.jpg'.format(name)
          cv2.imwrite(seamless_path, clones_without_seam)
        
    '''
        floof += 1
        if floof % 10000 == 0:
          print(floof)
        base = os.path.basename(img_fp)
        if(int(base.split("_")[0]) < RESUME_INDEX):
          continue
        if(base in output_paths):
          continue
        try:
          print(base)
          print(len(inputs))
          img_ori = cv2.imread(img_fp)
          if args.dlib_bbox:
              rects = face_detector(img_ori, 1)
          else:
              rects = []

          if len(rects) == 0:
              rects = dlib.rectangles()
              rect_fp = img_fp + '.bbox'
              lines = open(rect_fp).read().strip().split('\n')[1:]
              for l in lines:
                  l, r, t, b = [int(_) for _ in l.split(' ')[1:]]
                  rect = dlib.rectangle(l, r, t, b)
                  rects.append(rect)

          pts_res = []
          Ps = []  # Camera matrix collection
          poses = []  # pose collection, [todo: validate it]
          vertices_lst = []  # store multiple face vertices
          ind = 0
          suffix = get_suffix(img_fp)
          for rect in rects:
              # whether use dlib landmark to crop image, if not, use only face bbox to calc roi bbox for cropping
              if args.dlib_landmark:
                  # - use landmark for cropping
                  pts = face_regressor(img_ori, rect).parts()
                  pts = np.array([[pt.x, pt.y] for pt in pts]).T
                  roi_box = parse_roi_box_from_landmark(pts)
                  #print(roi_box)
              else:
                  # - use detected face bbox
                  bbox = [rect.left(), rect.top(), rect.right(), rect.bottom()]
                  roi_box = parse_roi_box_from_bbox(bbox)

              img = crop_img(img_ori, roi_box)
              img = cv2.resize(img, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
              input = transform(img).unsqueeze(0)
              imgs.append(img)
              inputs.append(input)
              input_paths.append(img_fp)
          if len(inputs) >= 1:
              print("DUMP")
              # forward: one step
              with torch.no_grad():
                  inputs = torch.cat(inputs)
                  if args.mode == 'gpu':
                      inputs = inputs.cuda()
                  param = model(inputs)
                  print(param.shape)
                  for i in range(param.shape[0]):
                    prediction = param[i].squeeze().cpu().numpy().flatten().astype(np.float32)
                    wfp_paf = '/media/rzats/Elements/UV_Data/Incomplete_UVs/{}'.format(os.path.basename(input_paths[i]))
                    #wfp_crop = '{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind)
                    paf_feature = gen_img_paf(img_crop=imgs[i], param=prediction, kernel_size=args.paf_size)
                    cv2.imwrite(wfp_paf, paf_feature)
              imgs = []
              inputs = []
              input_paths = []

              # 68 pts
              #pts68 = predict_68pts(param, roi_box)

              # two-step for more accurate bbox to crop face
              #if args.bbox_init == 'two':
              #    roi_box = parse_roi_box_from_landmark(pts68)
              #    img_step2 = crop_img(img_ori, roi_box)
              #    img_step2 = cv2.resize(img_step2, dsize=(STD_SIZE, STD_SIZE), interpolation=cv2.INTER_LINEAR)
              #    input = transform(img_step2).unsqueeze(0)
              #    with torch.no_grad():
              #        if args.mode == 'gpu':
              #            input = input.cuda()
              #        param = model(input)
              #        param = param.squeeze().cpu().numpy().flatten().astype(np.float32)
              #
              #    pts68 = predict_68pts(param, roi_box)

              #pts_res.append(pts68)
              #P, pose = parse_pose(param)
              #Ps.append(P)
              #poses.append(pose)

              # dense face 3d vertices
              #if args.dump_ply or args.dump_vertex or args.dump_depth or args.dump_pncc or args.dump_obj:
              #    vertices = predict_dense(param, roi_box)
              #    vertices_lst.append(vertices)
              #if args.dump_ply:
              #    dump_to_ply(vertices, tri, '{}_{}.ply'.format(img_fp.replace(suffix, ''), ind))
              #if args.dump_vertex:
              #    dump_vertex(vertices, '{}_{}.mat'.format(img_fp.replace(suffix, ''), ind))
              #if args.dump_pts:
              #    wfp = '{}_{}.txt'.format(img_fp.replace(suffix, ''), ind)
              #    np.savetxt(wfp, pts68, fmt='%.3f')
              #    print('Save 68 3d landmarks to {}'.format(wfp))
              #if args.dump_roi_box:
              #    wfp = '{}_{}.roibox'.format(img_fp.replace(suffix, ''), ind)
              #    np.savetxt(wfp, roi_box, fmt='%.3f')
              #    print('Save roi box to {}'.format(wfp))
              #if args.dump_paf:
              #    wfp_paf = '/media/rzats/Elements/UV_Data/Incomplete_UVs/{}'.format(os.path.basename(img_fp))
              #    wfp_crop = '{}_{}_crop.jpg'.format(img_fp.replace(suffix, ''), ind)
              #    paf_feature = gen_img_paf(img_crop=img, param=param, kernel_size=args.paf_size)

              #    cv2.imwrite(wfp_paf, paf_feature)
                  #cv2.imwrite(wfp_crop, img)
                  #print('Dump to {} and {}'.format(wfp_crop, wfp_paf))
                  #print('Dump to {}'.format(wfp_paf))
              #if args.dump_obj:
              #    wfp = '{}_{}.obj'.format(img_fp.replace(suffix, ''), ind)
              #    colors = get_colors(img_ori, vertices)
              #    write_obj_with_colors(wfp, vertices, tri, colors)
              #    print('Dump obj with sampled texture to {}'.format(wfp))
              #ind += 1

          #if args.dump_pose:
          #    # P, pose = parse_pose(param)  # Camera matrix (without scale), and pose (yaw, pitch, roll, to verify)
          #    img_pose = plot_pose_box(img_ori, Ps, pts_res)
          #    wfp = img_fp.replace(suffix, '_pose.jpg')
          #    cv2.imwrite(wfp, img_pose)
          #    print('z to {}'.format())
          #if args.dump_depth:
          #    wfp = img_fp.replace(suffix, '_depth.png')
          #    # depths_img = get_depths_image(img_ori, vertices_lst, tri-1)  # python version
          #    depths_img = cget_depths_image(img_ori, vertices_lst, tri - 1)  # cython version
          #    cv2.imwrite(wfp, depths_img)
          #    print('Dump to {}'.format(wfp))
          #if args.dump_pncc:
          #    wfp = img_fp.replace(suffix, '_pncc.png')
          #    pncc_feature = cpncc(img_ori, vertices_lst, tri - 1)  # cython version
          #    cv2.imwrite(wfp, pncc_feature[:, :, ::-1])  # cv2.imwrite will swap RGB -> BGR
          #    print('Dump to {}'.format(wfp))
          #if args.dump_res:
          #    draw_landmarks(img_ori, pts_res, wfp=img_fp.replace(suffix, '_3DDFA.jpg'), show_flg=args.show_flg)
        except KeyboardInterrupt:
          raise
        except:
          print("Unexpected error:", sys.exc_info()[0])
          with open("BAD", "a") as file_object:
            file_object.write(img_fp + "\n")
          pass     
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3DDFA inference pipeline')
    parser.add_argument('-f', '--files', nargs='+',
                        help='image files paths fed into network, single or multiple images')
    parser.add_argument('-m', '--mode', default='cpu', type=str, help='gpu or cpu mode')
    parser.add_argument('--show_flg', default='true', type=str2bool, help='whether show the visualization result')
    parser.add_argument('--bbox_init', default='one', type=str,
                        help='one|two: one-step bbox initialization or two-step')
    parser.add_argument('--dump_res', default='true', type=str2bool, help='whether write out the visualization image')
    parser.add_argument('--dump_vertex', default='false', type=str2bool,
                        help='whether write out the dense face vertices to mat')
    parser.add_argument('--dump_ply', default='true', type=str2bool)
    parser.add_argument('--dump_pts', default='true', type=str2bool)
    parser.add_argument('--dump_roi_box', default='false', type=str2bool)
    parser.add_argument('--dump_pose', default='true', type=str2bool)
    parser.add_argument('--dump_depth', default='true', type=str2bool)
    parser.add_argument('--dump_pncc', default='true', type=str2bool)
    parser.add_argument('--dump_paf', default='false', type=str2bool)
    parser.add_argument('--paf_size', default=3, type=int, help='PAF feature kernel size')
    parser.add_argument('--dump_obj', default='true', type=str2bool)
    parser.add_argument('--dlib_bbox', default='true', type=str2bool, help='whether use dlib to predict bbox')
    parser.add_argument('--dlib_landmark', default='true', type=str2bool,
                        help='whether use dlib landmark to crop image')

    args = parser.parse_args()
    main(args)
