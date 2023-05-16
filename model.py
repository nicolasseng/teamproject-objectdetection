import cv2
import numpy as np
import torch

model = torch.hub.load("ultralytics/yolov5", "yolov5s")

img = "data/sample_img/Image2.jpg"
results = model(img)
results.show()
