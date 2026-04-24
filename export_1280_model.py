# export_1280_feature_single.py
import torch
import torch.nn as nn
from torchvision import models

# 加载预训练模型
model = models.mobilenet_v2(pretrained=True)
model.eval()

# 提取特征层
class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.features = model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

feature_model = FeatureExtractor(model)
feature_model.eval()

# 测试输出
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = feature_model(dummy_input)
    print(f"输出维度: {output.shape}")

# 导出为单文件 ONNX（不使用外部数据）
torch.onnx.export(
    feature_model,
    dummy_input,
    "mobilenet_v2_1280_single.onnx",
    input_names=['input'],
    output_names=['feature'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'feature': {0: 'batch_size'}
    },
    opset_version=11,
    do_constant_folding=True,
    export_params=True,
    # 关键：禁止外部数据
    external_data=False
)

print("✅ 导出成功！单文件: mobilenet_v2_1280_single.onnx")

# 验证
import onnx
try:
    onnx_model = onnx.load("mobilenet_v2_1280_single.onnx")
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX 模型验证通过")
except Exception as e:
    print(f"⚠️ 验证失败: {e}")