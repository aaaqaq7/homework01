from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/v8n-zuoye-train3/weights/best.pt') # 自己训练结束后的模型权重
    model.val(data='ultralytics/cfg/datasets/zyoye.yaml',
              split='test',
              imgsz=640,
              batch=16,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='zuoye',
              )
