from yoloOpencv import opencvYOLO
import numpy as np
import cv2
import imutils
import time, sys
from PIL import ImageFont, ImageDraw, Image
from libGreen import webCam
from libGreen import OBJTracking

yolo = opencvYOLO(modeltype="yolov3", imgsize=(416,416), \
    objnames="obj.names", \
    weights="yolov3-tiny_90000.weights",\
    cfg="yolov3-tiny.cfg")

hot_area_x = (610, 1160)
count_line_x = 730

cam_rotate = 0
flip_vertical = False
flip_horizontal = False

frame_display_size = (800, 600)
video_file = "tv2.MOV"
write_video = False
output_rotate = False
rotate = 0

draw_face_box = True
cam_id = 0
webcam_size = (1920,1080)

output_video = "green02.avi"
frame_rate = 30

CAMERA = webCam(id=cam_id, videofile=video_file, size=webcam_size)
if(CAMERA.working() is False):
    print("webcam cannot work.")
    sys.exit()

CAMERA.set_record(outputfile=output_video, video_rate=frame_rate)
OB_TRACK = None

#fps count
start = time.time()
def fps_count(num_frames):
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    print("Estimated frames per second : {0}".format(fps))
    return fps

def printText(txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        ## Use cv2.FONT_HERSHEY_XXX to write English.
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "wt009.ttf"
        #print("TEST", txt)
        font = ImageFont.truetype(fontpath, int(size*20))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg

def findObject(frame):
    yolo.getObject(frame, labelWant="", drawBox=True, bold=2, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))

    return yolo.classIds, yolo.bbox, yolo.scores

def iou_bbox(box1, box2):
    #print(box1, box2)
    boxA = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
    boxB = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]

    boxA = [int(x) for x in boxA]
    boxB = [int(x) for x in boxB]

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def check_same(boxA, boxB):
    iou = iou_bbox(boxA, boxB)
    #print("IOU:", iou)
    if(iou>0.65):
        return True
    else:
        return False

def exit_app():
    print("End.....")
    CAMERA.stop_record()
    sys.exit(0)

frameID = 0
total_count = 0
re_recognize = True
ob_today_id = []
ob_today_bboxes = []
ob_today_classes = []

tracking_bboxes = []
need_tracking_bboxes = []
if __name__ == "__main__":
    
    hasFrame, frame_screen, frame_org = \
        CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal,\
            resize=(frame_display_size[0], frame_display_size[1]))

    bg = np.zeros((frame_org.shape[0], frame_org.shape[1], 3), np.uint8)
    bg[:] = (0, 255, 0)
    bg[0:1080,0:730] = (255,0,0)
    bg[0:1080,730:1160] = (0,0,255)
    #cv2.imshow("BG", bg)
    #print(bg.shape, frame_org.shape)

    while hasFrame:
        bbox_success, bbox_boxes, bbox_names, bbox_scores = [], [], [], []
        last_tracking_bboxes = need_tracking_bboxes

        ob_classes, ob_bboxes, ob_scores = findObject(frame_org.copy())
        #print("YOLO detect: ", ob_classes, ob_bboxes, ob_scores)

        need_tracking_bboxes, need_tracking_classes, need_tracking_scores = [], [], []
        for i, bbox in enumerate(ob_bboxes):
            cx, cy = bbox[0]+int(bbox[2]/2), bbox[1]+int(bbox[3]/2)
            point_color = bg[cy, cx]
            #print("point_color:", point_color[0], point_color[1], point_color[2])
            if(cx<hot_area_x[1] and cx>hot_area_x[0]):            
                cv2.rectangle(frame_org, (int(bbox[0]), int(bbox[1])), (int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])), (0,255,0), 2)
                need_tracking_bboxes.append(bbox)
                need_tracking_classes.append(ob_classes[i])
                need_tracking_scores.append(ob_scores[i])

            if(cx<count_line_x):
                frame_org = printText("TV #"+str(total_count), frame_org, color=(0,255,255,0), size=2.0, pos=(bbox[0]+int(bbox[2]/2),bbox[1]-30), type="Chinese")


        #YOLOBBOX與目前的 Tracking結果比較，有多的表示為新增ROI。
        new_object = False
        for i, new_bbox in enumerate(need_tracking_bboxes):
            exists = False
            for ii, tracking_bbox in enumerate(last_tracking_bboxes):
                #如果目前偵測出的物件與上一個影格追蹤的物件為同一個，則檢查有沒有通過計數線（一個為紅，一個為blue）
                #print("Check same:", new_bbox, tracking_bbox)
                if(check_same(new_bbox, tracking_bbox)):
                    exists = True
                    now_color = bg[new_bbox[1]+int(new_bbox[3]/2), new_bbox[0]+int(new_bbox[2]/2)]
                    last_color = bg[tracking_bbox[1]+int(tracking_bbox[3]/2), tracking_bbox[0]+int(tracking_bbox[2]/2)]
                    print("last_color:{}, now_color:{}".format(last_color, now_color))

                    if((last_color[0]==0 and last_color[1]==0 and last_color[2]==255) and (now_color[0]==255 and now_color[1]==0 and now_color[2]==0)):
                        total_count += 1
                        cv2.rectangle(frame_org, (int(new_bbox[0]), int(new_bbox[1])), (int(new_bbox[0]+new_bbox[2]), int(new_bbox[1]+new_bbox[3])), (0,255,255), 4)

            if(exists==False):  #如果exists一直為False, 則代表該new_bbox為新增的
                new_object = True

        '''
        if(new_object is True):
            print("Found new object.")
            OB_TRACK = OBJTracking()
            OB_TRACK.setROIs(frame_org, new_bbox, "KCF")


        if(OB_TRACK is not None):
            print("     Tracking......")
            (success, tracking_bboxes) = OB_TRACK.trackROI(frame_org)
        '''
        frame_org = printText("總計："+str(total_count), frame_org, color=(255,255,255,0), size=2.25, pos=(50,30), type="Chinese")
        cv2.imshow("FRAME", imutils.resize(frame_org, width=frame_display_size[0]))
        #cv2.imshow("BG", imutils.resize(bg, width=frame_display_size[0]))

        CAMERA.write_video(frame_org)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            exit_app()

        hasFrame, frame_screen, frame_org = \
            CAMERA.getFrame(rotate=cam_rotate, vflip=flip_vertical, hflip=flip_horizontal, resize=(frame_display_size[0], frame_display_size[1]))


        frameID += 1

    exit_app()
