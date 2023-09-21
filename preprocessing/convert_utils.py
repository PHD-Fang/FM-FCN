import cv2
import numpy as np


class WrapperCap:
    def __init__(self, videos):
        self.m_videos = videos
        self.m_index = 0
        self.m_width = videos.shape[2]
        self.m_height = videos.shape[1]
        self.m_frame_num = videos.shape[0]

    def read(self):
        frame = None
        ret = False
        if self.m_index < self.m_frame_num:
            frame = self.m_videos[self.m_index]
            self.m_index += 1
            ret = True
        return ret, frame

    def get(self, flag):
        if flag == cv2.CAP_PROP_FRAME_WIDTH:
            return self.m_width
        elif flag == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.m_height

def getEuclideanDistance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def getCenterBox(boxes, width, height):
    '''
    获取最居中的框
    Args:
        boxes:
        width:
        height:

    Returns:

    '''
    best_box = []
    best_distance = width * height
    center_x = width / 2
    center_y = height / 2
    for box in boxes:
        x1, y1, x2, y2 = box
        distance = getEuclideanDistance((x1 +x2) / 2, (y1 + y2) / 2, center_x, center_y)
        if distance < best_distance:
            best_distance = distance
            best_box = box
    return best_box

def getLocFromVideo(cap, fd):
    '''
    进行人脸检测，获取人脸框
    Args:
        cap:
        fd:

    Returns:

    '''

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))      # 视频图像宽度
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    # 视频图像高度
    face_boxes = []
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        boxes, _, _ = fd.getFaceLoc(frame)
        if len(boxes) == 0:
            face_boxes.append({
                'success': False,
                'box': [0, 0, 0, 0]
            })
        else:
            if len(boxes) > 1:
                box = getCenterBox(boxes, width, height)
            else:
                box = boxes[0]
            face_boxes.append({
                'success': True,
                'box': box
            })
    return face_boxes

def getFaceList(cap, face_boxes, img_size, frame_total):
    # 确定人脸区域尺寸
    for box in face_boxes:
        if box['success']:
            prev_box = np.array(box['box'])
            x1, y1, x2, y2 = prev_box
            minx = np.min((x1, x2))
            maxx = np.max((x1, x2))
            miny = np.min((y1, y2))
            maxy = np.max((y1, y2))

            # y_range_ext = (maxy - miny) * 0.2
            # miny -= y_range_ext

            cnt_x = np.round((minx + maxx) / 2).astype('int')
            cnt_y = np.round((maxy + miny) / 2).astype('int')
            break
    box_size = np.round(1. * (maxy - miny)).astype('int')

    raw_video = np.empty((frame_total, img_size, img_size, 3))
    for curr_frame in range(frame_total):
        box = face_boxes[curr_frame]
        if box['success']:
            curr_box = np.array(box['box'])
            curr_box = curr_box * 0.2 + prev_box * 0.8
            prev_box = curr_box
            x1, y1, x2, y2 = curr_box
            minx = np.min((x1, x2))
            maxx = np.max((x1, x2))
            miny = np.min((y1, y2))
            maxy = np.max((y1, y2))
            # y_range_ext = (maxy - miny) * 0.2
            # miny -= y_range_ext

            cnt_x = np.round((minx + maxx) / 2).astype('int')
            cnt_y = np.round((maxy + miny) / 2).astype('int')

        curr_box_size = np.round(1. * (maxy - miny)).astype('int')
        box_size = int(curr_box_size * 0.2 + box_size * 0.8)
        ret, frame = cap.read()
        assert ret, "Can't receive frame. Exiting ..."

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ########## for bbox ################
        box_half_size = int(box_size / 2)

        face = np.take(frame, range(cnt_y - box_half_size, cnt_y - box_half_size + box_size), 0,
                       mode='clip')
        face = np.take(face, range(cnt_x - box_half_size, cnt_x - box_half_size + box_size), 1,
                       mode='clip')

        if img_size == box_size:
            raw_video[curr_frame] = face
        else:
            raw_video[curr_frame] = cv2.resize(face, (img_size, img_size), cv2.INTER_AREA)
    return raw_video


