import argparse
from pathlib import Path

import os
import copy
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import time
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import numpy as np
confidence = 0.5
# 全局變量區域
left_arm_curl_count = 0
right_arm_curl_count = 0
left_arm_reset = 0
right_arm_reset = 0
left_arm_angle_prev = None
right_arm_angle_prev = None
time_prev = None

def calculate_angle(a, b, c):
    """計算由三點(a, b, c)形成的夾角"""
    ab = np.subtract(b, a)
    bc = np.subtract(c, b)
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_angular_velocity(angle_prev, angle_curr, time_diff):
    if angle_prev is not None and time_diff > 0:
        return (angle_curr - angle_prev) / time_diff
    return 0

def detect_hand_raise(kpts):
    global left_arm_angle_prev, right_arm_angle_prev, time_prev
    global left_arm_curl_count, right_arm_curl_count, left_arm_reset, right_arm_reset

    current_time = time.time()
    if time_prev is None:
        time_prev = current_time

    time_diff = current_time - time_prev

    kpts = [float(kpt) for kpt in kpts]
    left_shoulder = np.array([kpts[15], kpts[16]])
    left_elbow = np.array([kpts[21], kpts[22]])
    left_wrist = np.array([kpts[27], kpts[28]])
    right_shoulder = np.array([kpts[18], kpts[19]])
    right_elbow = np.array([kpts[24], kpts[25]])
    right_wrist = np.array([kpts[30], kpts[31]])

    # 計算左手臂夾角
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    left_arm_velocity = calculate_angular_velocity(left_arm_angle_prev, left_arm_angle, time_diff)
    left_arm_angle_prev = left_arm_angle

    if kpts[17] > confidence and kpts[23] > confidence and kpts[29] > confidence:
        if left_arm_angle <= 15:
            left_arm_reset = 1
        if left_arm_angle > 120 and left_arm_reset == 1:
            left_arm_curl_count += 1
            left_arm_reset = 0

    # 計算右手臂夾角
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    right_arm_velocity = calculate_angular_velocity(right_arm_angle_prev, right_arm_angle, time_diff)
    right_arm_angle_prev = right_arm_angle

    if kpts[20] > confidence and kpts[26] > confidence and kpts[32] > confidence:
        if right_arm_angle <= 15:
            right_arm_reset = 1
        if right_arm_angle > 120 and right_arm_reset == 1:
            right_arm_curl_count += 1
            right_arm_reset = 0

    time_prev = current_time

    return left_arm_angle, left_arm_velocity, right_arm_angle, right_arm_velocity

def display_hand_raise_status(im0, kpts):
    kpts = [float(kpt) for kpt in kpts]
    left_arm_angle, left_arm_velocity, right_arm_angle, right_arm_velocity = detect_hand_raise(kpts)
    
    cv2.putText(im0, f"Left Arm Velocity: {left_arm_velocity:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Right Arm Velocity: {right_arm_velocity:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # 假設鼻子的索引是 0 (鼻子的 x 和 y)
    nose_x = int(kpts[0])
    nose_y = int(kpts[1])
    left_eye = np.array([kpts[3], kpts[4]])
    right_eye = np.array([kpts[6], kpts[7]])
    eye_midpoint = (left_eye + right_eye) / 2
    #計算眼睛中點到鼻子的距離
    distance = int(np.sqrt((eye_midpoint[0] - nose_x) ** 2 + (eye_midpoint[1] - nose_y) ** 2))

    # 設置位置為鼻子上方一定距離
    text_position = (nose_x - 40, nose_y - distance * 5)  # 在鼻子上方 眼睛到鼻子五倍距離 像素處顯示文本
    status = detect_hand_raise(kpts)[0]
    # 在圖像上顯示舉手狀態
    cv2.putText(im0, str(status), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

    # 顯示左手臂彎曲次數
    cv2.putText(im0, f"Left Arm Curls: {left_arm_curl_count}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    # 顯示右手臂彎曲次數
    cv2.putText(im0, f"Right Arm Curls: {right_arm_curl_count}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

def find_neck_position(kpts):
    kpts = [float(kpt) for kpt in kpts]
    left_ear = np.array([kpts[9], kpts[10]])
    right_ear = np.array([kpts[12], kpts[13]])
    left_shoulder = np.array([kpts[15], kpts[16]])
    right_shoulder = np.array([kpts[18], kpts[19]])

    ear_midpoint = (left_ear + right_ear) / 2
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    neck_position = (ear_midpoint + shoulder_midpoint) / 2
    return tuple(neck_position.astype(int))

def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)

    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'

    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())

    if half:
        model.half()

    cudnn.benchmark = True
    dataset = LoadStreams(source, img_size=imgsz) if webcam else LoadImages(source, img_size=imgsz)

    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)

        for i, det in enumerate(pred):
            if webcam:
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            save_path = str(save_dir / Path(p).name)
            txt_path = str(save_dir / 'labels' / Path(p).stem) + ('' if dataset.mode == 'image' else f'_{frame}')
            s += '%gx%g ' % img.shape[2:]
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
            imc = im0.copy() if opt.save_crop else im0

            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()
                    s += f'{n} {names[int(c)]}s, '

                for *xyxy, conf, cls in reversed(det):
                    if save_txt:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if opt.save_txt_tidl:
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                        with open(txt_path + '_tidl.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=opt.line_thickness)

                        kpts = det[:, 5:].view(-1, 3).t().cpu().numpy()
                        for kid, kpt in enumerate(kpts):
                            x_coord, y_coord, conf = int(kpt[0]), int(kpt[1]), kpt[2]
                            if conf > opt.kpt_conf_thres:
                                cv2.circle(im0, (x_coord, y_coord), 4, (0, 0, 255), -1)
                                cv2.putText(im0, f'{kid}', (x_coord, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        if kpts.size:
                            display_hand_raise_status(im0, kpts.flatten().tolist())

            print(f'{s}Done. ({time.time() - t0:.3f}s)')

            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)

            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()
                        if vid_cap:
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt for TIDL deployment')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    parser.add_argument('--kpt-label', action='store_true', help='use keypoint label')
    parser.add_argument('--fourcc', default='mp4v', help='fourcc')
    parser.add_argument('--kpt-conf-thres', type=float, default=0.5, help='Keypoint confidence threshold')

    opt = parser.parse_args()
    opt.img_size = check_img_size(opt.img_size)
    print(opt)

    with torch.no_grad():
        detect(opt)
