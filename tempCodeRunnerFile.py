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
confidence=0.5
# 全局變量區域
left_arm_curl_count = 0
right_arm_curl_count = 0
left_arm_reset = 0
right_arm_reset = 0
left_arm_velocity = 0
right_arm_velocity = 0
left_arm_angle_prev = 0
right_arm_angle_prev = 0
time_prev = None  # 初始化为 None
accumulated_time = 0 # 累积时间
stand_up_count=0
status = ""
last_status = ""
standing_knee_flex_count = 0
standing_knee_flex_status=""
LeftKneeToHip_Distance=0
RightKneeToHip_Distance=0
stand_stright_already=False
def calculate_angle(a, b, c):
    """計算由三點(a, b, c)形成的夾角"""
    ab = np.subtract(b, a)
    bc = np.subtract(c, b)
    cosine_angle = np.dot(ab, bc) / (np.linalg.norm(ab) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

def calculate_angular_velocity(angle_start, angle_end, time_period):
    if time_period > 0:
        return (angle_end - angle_start) / time_period
    return 0

def detect_hand_raise(kpts):
    global left_arm_angle_prev, right_arm_angle_prev, time_prev, accumulated_time
    global left_arm_velocity, right_arm_velocity
    global left_arm_curl_count, right_arm_curl_count, left_arm_reset, right_arm_reset
    
    current_time = time.time()
    if time_prev is None:
        time_prev = current_time

    time_diff = current_time - time_prev
    accumulated_time += time_diff
    
    #定義左肩膀、左肘、左手腕、右肩膀、右肘和右手腕的坐標
    kpts = [float(kpt) for kpt in kpts]
    left_shoulder = np.array([kpts[15], kpts[16]])
    left_elbow = np.array([kpts[21], kpts[22]])
    left_wrist = np.array([kpts[27], kpts[28]])
    right_shoulder = np.array([kpts[18], kpts[19]])
    right_elbow = np.array([kpts[24], kpts[25]])
    right_wrist = np.array([kpts[30], kpts[31]])
    
    
     # 計算手臂夾角
    left_arm_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_arm_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # 偵測左手臂彎曲
    if accumulated_time >= 1:  # 每秒计算一次角速度
        left_arm_velocity = calculate_angular_velocity(left_arm_angle_prev, left_arm_angle, accumulated_time)
        right_arm_velocity = calculate_angular_velocity(right_arm_angle_prev, right_arm_angle, accumulated_time)
        left_arm_angle_prev = left_arm_angle
        right_arm_angle_prev = right_arm_angle
        accumulated_time = 0  # 重置累积时间
        
    
    #定義手臂彎取為15度到120度一次
    if kpts[17]>confidence and kpts[23]>confidence and kpts[29]>confidence:
        if left_arm_angle <=15:
            left_arm_reset=1
        if left_arm_angle > 120 and left_arm_reset==1:
            left_arm_curl_count += 1
            left_arm_reset=0
    

    
    # 偵測右手臂彎曲
    if kpts[20]>confidence and kpts[26]>confidence and kpts[32]>confidence:
        if right_arm_angle <=15:
            right_arm_reset=1
        if right_arm_angle > 120 and right_arm_reset==1:
            right_arm_curl_count += 1
            right_arm_reset=0
    time_prev = current_time            
    #print('left_arm_angle='+str(left_arm_angle))
    #print(" "+str(left_arm_reset))
    #print('right_arm_angle='+str(right_arm_angle))
    #print(" "+str(right_arm_reset))
    #print("當前角度="+str(right_arm_angle))
    #print("上個角度="+str(right_arm_angle_prev))
    
    #手腕要高於肩膀 手肘要高於肩膀 
    if(right_wrist[1] < right_shoulder[1] and
        right_elbow[1] < right_shoulder[1]and
        left_wrist[1] < left_shoulder[1] and
        left_elbow[1] < left_shoulder[1]and
        kpts[17]>confidence and kpts[23]>confidence and kpts[29]>confidence and kpts[20]>confidence and kpts[26]>confidence and kpts[32]>confidence):
        return "raise two hands"
    if (left_wrist[1] < left_shoulder[1] and
        left_elbow[1] < left_shoulder[1]and
        kpts[17]>confidence and kpts[23]>confidence and kpts[29]>confidence):
        return "raise left hand"
    if (right_wrist[1] < right_shoulder[1] and
        right_elbow[1] < right_shoulder[1]
        and kpts[20]>confidence and kpts[26]>confidence and kpts[32]>confidence):
        return "raise right hand"    
    return "no raise hand"

#顯示舉手狀態
def display_hand_raise_status(im0, kpts):
    kpts = [float(kpt) for kpt in kpts]
    status = detect_hand_raise(kpts)
    cv2.putText(im0, f"Left Arm Velocity: {left_arm_velocity:.2f}", (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Right Arm Velocity: {right_arm_velocity:.2f}", (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    if status != "no raise hand":
        # 假設鼻子的索引是 0 (鼻子的 x 和 y)
        nose_x = int(kpts[0])
        nose_y = int(kpts[1])
        left_eye=np.array([kpts[3],kpts[4]])
        right_eye=np.array([kpts[6],kpts[7]])
        eye_midpoint = (left_eye+right_eye)/2
        #計算眼睛中點到鼻子的距離
        distance =int(np.sqrt((eye_midpoint[0] - nose_x) ** 2 + (eye_midpoint[1] - nose_y) ** 2))

        # 設置位置為鼻子上方一定距離
        text_position = (nose_x-40, nose_y - distance*5)  # 在鼻子上方 眼睛到鼻子五倍距離 像素處顯示文本

        # 在圖像上顯示舉手狀態
        cv2.putText(im0, status, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
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
#判斷站直

def detect_stand_sit(kpts):
    kpts = [float(kpt) for kpt in kpts]
    global stand_up_count,status,last_status
    
    
    left_hip = np.array([kpts[33], kpts[34]])
    right_hip = np.array([kpts[36], kpts[37]])
    mid_hip=(left_hip+right_hip)/2

    left_knee = np.array([kpts[39], kpts[40]])
    right_knee = np.array([kpts[42], kpts[43]])
    mid_knee = (left_knee+right_knee)/2
    
    
    if mid_hip[1]<mid_knee[1]-np.linalg.norm(left_hip-right_hip):
        
        status="stand up"
    else:
        status="sit down"
        if last_status=="stand up":
            stand_up_count += 1
    last_status=status  
          
def is_standing_straight(im0,kpts):
    # 提取關鍵點坐標
    kpts = [float(kpt) for kpt in kpts]
    global LeftKneeToHip_Distance,RightKneeToHip_Distance,stand_stright_already
    #判斷置信度低於預設值函式返回
    parameters = [kpts[17],kpts[20],kpts[35],kpts[38],kpts[41],kpts[44],kpts[47],kpts[50]]
    for param in parameters:
        if param < confidence:
            return False
    
    #定義關鍵點
    
    left_shoulder = np.array([kpts[15], kpts[16]])
    right_shoulder = np.array([kpts[18], kpts[19]])
    left_hip = np.array([kpts[33], kpts[34]])
    right_hip = np.array([kpts[36], kpts[37]])
    left_knee = np.array([kpts[39], kpts[40]])
    right_knee = np.array([kpts[42], kpts[43]])
    left_ankle = np.array([kpts[45], kpts[46]])
    right_ankle = np.array([kpts[48], kpts[49]])
    
    shoulders = np.array([left_shoulder, right_shoulder])
    hips = np.array([left_hip, right_hip])
    knees = np.array([left_knee, right_knee])
    
    # 垂直對齊檢查
    def is_vertical(a, b, tolerance=20):
        return abs(a[0] - b[0]) < tolerance
    
    # 水平對齊檢查
    def is_horizontal(a, b, tolerance=20):
        return abs(a[1] - b[1]) < tolerance
    
    

    # 肩膀和髖部的垂直對齊
    vertical_check_1 = is_vertical(shoulders[0], hips[0])
    vertical_check_2 = is_vertical(shoulders[1], hips[1])

    # 髖部和膝蓋的垂直對齊
    vertical_check_3 = is_vertical(hips[0], knees[0])
    vertical_check_4 = is_vertical(hips[1], knees[1])
    
     # 肩膀和髖部的水平對齊
    horizontal_check_1 = is_horizontal(shoulders[0], shoulders[1])
    horizontal_check_2 = is_horizontal(hips[0], hips[1])
    horizontal_check_3 = is_horizontal(knees[0], knees[1])
    
    
    

    # 檢查所有條件
    if ( vertical_check_1 and vertical_check_2 and vertical_check_3 and vertical_check_4
        and horizontal_check_1 and horizontal_check_2 and horizontal_check_3 ):
        cv2.putText(im0, f"ReadyToKneeFlex", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 1)
        stand_stright_already=True
        #print("true")
        LeftKneeToHip_Distance = np.linalg.norm(left_knee - left_hip)
        RightKneeToHip_Distance = np.linalg.norm(right_knee - right_hip)
        return 
    
#站立屈膝
def standing_knee_flexion(im0,kpts):
    global  standing_knee_flex_count,standing_knee_flex_status
    kpts = [float(kpt) for kpt in kpts]
    parameters = [kpts[17],kpts[20],kpts[35],kpts[38],kpts[41],kpts[44],kpts[47],kpts[50]]
    
    for param in parameters:
        if param < confidence:
            cv2.putText(im0, f"Cannot find specific keypoints !!", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
            return False
        
    cv2.putText(im0, f"standing knee flex Count: {standing_knee_flex_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    left_hip = np.array([kpts[33], kpts[34]])
    right_hip = np.array([kpts[36], kpts[37]])
    left_knee = np.array([kpts[39], kpts[40]])
    right_knee = np.array([kpts[42], kpts[43]])
    left_ankle = np.array([kpts[45], kpts[46]])
    right_ankle = np.array([kpts[48], kpts[49]])
    
    if stand_stright_already==False:
        return
        
    
    
    if left_knee[1]<left_hip[1]+LeftKneeToHip_Distance*0.67:
        standing_knee_flex_status="left_knee_up"
        cv2.putText(im0, f"Left Knee Up", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    elif right_knee[1]<right_hip[1]+RightKneeToHip_Distance*0.67:
        cv2.putText(im0, f"Right Knee Up", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if standing_knee_flex_status=="left_knee_up":
            standing_knee_flex_count+=1
        standing_knee_flex_status="right_knee_up"
    
    
    
def display_stand_sit_status(im0,kpts):
    kpts = [float(kpt) for kpt in kpts]
    detect_stand_sit(kpts)
    cv2.putText(im0, f"Stand Up Count: {stand_up_count}", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(im0, f"Status: {status}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    
def detect_stand_up(kpts):
    
    
    kpts = [float(kpt) for kpt in kpts]
    left_hip = np.array([kpts[33], kpts[34]])
    right_hip = np.array([kpts[36], kpts[37]])
    left_knee=np.array([kpts[39],kpts[40]])
    right_knee=np.array([kpts[42],kpts[43]])
    #預備動作站立時判斷膝蓋到臀部的中間距離
    if status == "stand up":
        Knee_Lift=np.linalg.norm(left_knee-left_hip)/2
        
            
    last_status = status

def display_stand_up_status(im0, kpts):
    detect_stand_up(kpts)
    cv2.putText(im0, f"Stand Up: {stand_up_count}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    

    
def detect(opt):
    source, weights, view_img, save_txt, imgsz, save_txt_tidl, kpt_label = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_txt_tidl, opt.kpt_label
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok)  # increment run
    (save_dir / 'labels' if (save_txt or save_txt_tidl) else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu' and not save_txt_tidl  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    if isinstance(imgsz, (list,tuple)):
        assert len(imgsz) ==2; "height and width of image has to be specified"
        imgsz[0] = check_img_size(imgsz[0], s=stride)
        imgsz[1] = check_img_size(imgsz[1], s=stride)
    else:
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    names = model.module.names if hasattr(model, 'module') else model.names  # get class names
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
    
    #主要檢測函數
    for path, img, im0s, vid_cap in dataset:
        # 開始時間
        frame_time_start = time.time()
        
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        print(pred[...,4].max())
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms, kpt_label=kpt_label)
        t2 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)
        
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                scale_coords(img.shape[2:], det[:, :4], im0.shape, kpt_label=False)
                scale_coords(img.shape[2:], det[:, 6:], im0.shape, kpt_label=kpt_label, step=3)

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(det[:,:6])):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or opt.save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if opt.hide_labels else (names[c] if opt.hide_conf else f'{names[c]} {conf:.2f}')
                        kpts = det[det_index, 6:]
                        plot_one_box(xyxy, im0, label=label, color=colors(c, True), line_thickness=opt.line_thickness, kpt_label=kpt_label, kpts=kpts, steps=3, orig_shape=im0.shape[:2])
                        #標記每個節點
                        for j in range(0, len(kpts), 3):
                            if kpts[j+2]> confidence:
                                cv2.putText(im0, f"{j//3}", (int(kpts[j]), int(kpts[j+1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
                        #計算脖子的位子
                        neck_x, neck_y = find_neck_position(kpts)
                        cv2.putText(im0, "17", (neck_x, neck_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1, cv2.LINE_AA)
                        #舉手演算法
                        #display_hand_raise_status(im0, kpts)
                        #display_stand_sit_status(im0,kpts)
                        standing_knee_flexion(im0,kpts)
                        is_standing_straight(im0,kpts)
                        
                        if opt.save_crop:
                            save_one_box(xyxy, im0s, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                            
                        


                if save_txt_tidl:  # Write to file in tidl dump format
                    for *xyxy, conf, cls in det_tidl:
                        xyxy = torch.tensor(xyxy).view(-1).tolist()
                        line = (conf, cls,  *xyxy) if opt.save_conf else (cls, *xyxy)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

            # Print time (inference + NMS)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            # 更新 FPS
            frame_time_end = time.time()
            time_diff = frame_time_end - frame_time_start
            if time_diff>0:
                fps = 1 / time_diff
                
                cv2.putText(im0, f"FPS: {fps:.2f}", (5, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 1)

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)
           

    if save_txt or save_txt_tidl or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt or save_txt_tidl else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/videos', help='source')  # file/folder, 0 for webcam data/images
    parser.add_argument('--img-size', nargs= '+', type=int, default=360, help='inference size (pixels)')#原640
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-txt-tidl', action='store_true', help='save results to *.txt in tidl format')
    parser.add_argument('--save-bin', action='store_true', help='save base n/w outputs in raw bin format')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--kpt-label', default=True, help='use keypoint labels')
    opt = parser.parse_args()
    print(opt)
    check_requirements(exclude=('tensorboard', 'pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect(opt=opt)
                strip_optimizer(opt.weights)
        else:
            detect(opt=opt)
