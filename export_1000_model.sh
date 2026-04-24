#!/bin/bash

# 下载MobileNetV2 ONNX模型（约13MB）
wget -O src/main/resources/models/mobilenet_v2.onnx \
  https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx

echo "Model downloaded successfully!"