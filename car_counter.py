from ultralytics import YOLO
import cv2
from sort import*
import cvzone
import math



mymodel = YOLO('yolov8m.pt')



coco_classes = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
    'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]




cap = cv2.VideoCapture("Cars Moving On Road Stock Footage - Free Download.mp4")

cap.set(3, 1280)  # Frame genişliğini ayarla
cap.set(4, 720)  # Frame yüksekliğini ayarla




mask = cv2.imread("region.png")
mask = cv2.resize(mask, (1280, 720))  # mask boyutunu frame ile aynı boyuta getir

tracker = Sort(max_age=15,min_hits=1, iou_threshold=0.3)

limit =[312,240,1061,240]

totalCount=[]
while True:
    ret, frame = cap.read()

    imgRegion = cv2.bitwise_and(frame, mask)
    results = mymodel(imgRegion, stream=True)

    detections=np.empty((0,5 ))
   
    for r in results:
        boxes =r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            currentClass = coco_classes[cls]
            
            
            if currentClass == "car" or  currentClass == "truck" or\
            currentClass =="bus":
                
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections,currentArray))

    confThreshold = 0.3
    nmsThreshold = 0.5
    indices = cv2.dnn.NMSBoxes(detections[:, :4], detections[:, 4], confThreshold, nmsThreshold)
    filtered_detects = detections[indices]         
    
    resultsTracker = tracker.update(filtered_detects)
    cv2.line(frame,(limit[0],limit[1]),(limit[2],limit[3]),(0,0,255),3)
    for result in resultsTracker:
        x1,y1,x2,y2,id =result
        x1,y1,x2,y2  = int(x1),int(y1),int(x2),int(y2)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        w,h = x2-x1,y2-y1
        cx,cy = x1+w//2,y1+h//2
        cv2.circle(frame,(int(cx),int(cy)),2,(255,0,255),cv2.FILLED)
        
       
        


        if limit[0]< cx <limit[2] and limit[1]-10 <cy< limit[2] +10 :
            if totalCount.count(id)==0 or len(totalCount)==0:
                totalCount.append(id)
                cv2.line(frame,(limit[0],limit[1]),(limit[2] ,limit[3]),(0,255,0),5)

        # cv2.putText(frame, f'Araba sayisi = {len(totalCount)}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3 )
    cvzone.putTextRect(frame, f'Araba sayisi = {len(totalCount)}', (50,50))
        
     
    cv2.imshow("screen", frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



  

   
# Tüm pencereleri kapatmak ve kaynakları serbest bırakmak için
cv2.destroyAllWindows()
cap.release()   
    