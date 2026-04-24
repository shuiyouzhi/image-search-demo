# Start: 2026-04-23
## “以图搜图”功能. resources/models目录下有3个模型. 用于查询本地图库中相似产品

## 模型选择. 执行命令下载相关模型. sh: 直接运行. py: python *.py
export_1000_model.sh 下载维度1000的模型, 13MB  
export_1280_model.py 下载维度1280的模型, 13MB  
export_CLIP_model.py 下载维度CLIP维度512的模型(ViT-B/32), 600MB
## 3个模型区别
1000: 部分轻量分类模型,通用商品识别,介于轻量和精细之间. 能搜出不同款的产品  
1280: MobileNetV2/EfficientNet, 平衡之选：效果好且速度快, 移动端实时检索、通用图搜. 能搜出不同款的产品  
CLIP: 维度512. 多模态对齐能力强，速度快. 图像-文本对齐, 训练数据多样, 对颜色变化不敏感, 能理解“同款不同色”的语义. 通常选择相似度在85%以上的产品作为同一个产品  
***具体请自行验证后选取合适的模型***

## 其他可用的CLIP模型变体
#### 如果 ViT-B/32 模型太大，也可以使用其他版本：
| 模型名称   | 输出维度 | 大小   | 说明   |
|--------|-----|--------|--------|
| ViT-B/32   | 512  | ~600MB   |平衡之选   |
| ViT-B/16   | 512  | ~600MB   |精度稍高，速度稍慢   |
| RN50   | 1024  | ~300MB   |ResNet架构，速度更快  |
| RN101   | 512  | ~400MB   |ResNet架构，精度更高   |

## 向量数据库：Milvus
### 安装方式:
根目录下执行docker compose up -d  
数据库运维地址: http://localhost:9091/webui  
数据库管理地址: http://localhost:8011  

## 接口: ImageSearchController
上传图片: uploadImage  
指定路径批量上传图片: uploadByPath  
图搜: searchSimilar  
