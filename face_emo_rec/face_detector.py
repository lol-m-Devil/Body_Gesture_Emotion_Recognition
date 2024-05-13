import common
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
from model.DBFace import DBFace

from PIL import Image

HAS_CUDA = torch.cuda.is_available()
print(f"HAS_CUDA = {HAS_CUDA}")

def intv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)



def nms(objs, iou=0.5):

    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):

        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


def detect(model, image, threshold=0.4, nms_iou=0.5):

    mean = [0.408, 0.447, 0.47]
    std = [0.289, 0.274, 0.278]

    image = common.pad(image)
    image = ((image / 255.0 - mean) / std).astype(np.float32)
    image = image.transpose(2, 0, 1)

    torch_image = torch.from_numpy(image)[None]
    if HAS_CUDA:
        torch_image = torch_image.cuda()

    hm, box, landmark = model(torch_image)
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


        

def face_detector(file):

    dbface = DBFace()
    dbface.eval()

    if HAS_CUDA:
        dbface.cuda()

    dbface.load("face_emo_rec/model/dbface.pth")
    cap = cv2.VideoCapture(file)
    ok, frame = cap.read()
    keyframes = []
    while ok:
        objs = detect(dbface, frame)
        if(len(objs) != 0):
            left, top, right, bottom = intv(objs[0].box)
            cropped_frame = frame[top:bottom, left:right]
            cropped_frame = cv2.resize(cropped_frame, (224,224))
            # cv2.imshow("demo DBFace", cropped_frame)
            frameTensor = torch.tensor(cropped_frame, dtype=torch.float32).view(3, *(224, 224))
            keyframes.append(frameTensor)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        ok, frame = cap.read()
    
    cap.release()
    cv2.destroyAllWindows()
    return keyframes
    

    
    
    


    