# Detection Toolbox
1. [MMDetection](https://github.com/open-mmlab/mmdetection)
2. [MMCV](https://github.com/open-mmlab/mmcv)
3. [Detectron](https://github.com/facebookresearch/Detectron/)
4. [Detectron2](https://github.com/facebookresearch/detectron2)
5. [SimpleDet](https://github.com/TuSimple/simpledet)
6. [Maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

# Paper list

## Datasets
1. **[PASCAL VOC]** The PASCAL Visual Object Classes (VOC) Challenge | **[IJCV' 10]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf)

2. **[PASCAL VOC]** The PASCAL Visual Object Classes Challenge: A Retrospective | **[IJCV' 15]** | [`[pdf]`](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham15.pdf) | [`[link]`](http://host.robots.ox.ac.uk/pascal/VOC/)

3. **[COCO]** Microsoft COCO: Common Objects in Context | **[ECCV' 14]** | [`[pdf]`](https://arxiv.org/pdf/1405.0312.pdf) | [`[link]`](http://cocodataset.org/)

4. **[Open Images]** The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale | **[arXiv' 18]** | [`[pdf]`](https://arxiv.org/pdf/1811.00982v1.pdf) | [`[link]`](https://storage.googleapis.com/openimages/web/index.html)


## Object Detection

### two-stage

1. **[R-CNN]** Rich feature hierarchies for accurate object detection and semantic segmentation [`[pdf]`](https://arxiv.org/pdf/1311.2524)[`[official code - caffe]`](https://github.com/rbgirshick/rcnn)

2. **[Fast R-CNN]** Fast R-CNN  [`[pdf]`](https://arxiv.org/pdf/1504.08083)[`[official code - caffe]`](https://github.com/rbgirshick/fast-rcnn)

3. **[Faster R-CNN]** Faster R-CNN  [`[pdf]`](https://arxiv.org/pdf/1506.01497) [`[official code - caffe]`](https://github.com/rbgirshick/py-faster-rcnn)[`[code - pytorch]`](https://github.com/chenyuntc/simple-faster-rcnn-pytorch)

### one-stage

1. **[YOLO-v1]** You Only Look Once: Unified, Real-Time Object Detection [`[pdf]`](https://arxiv.org/pdf/1506.02640) [`[code - tensorflow]`](https://github.com/JunshengFu/vehicle-detection) [`[code - pytorch]`](https://github.com/xiongzihua/pytorch-YOLO-v1)

2. **[YOLO-v2]** YOLO9000: Better, Faster, Stronger [`[pdf]`](https://arxiv.org/pdf/1612.08242) [`[code - pytorch]`](https://github.com/longcw/yolo2-pytorch)

3. **[YOLO-v3]** YOLOv3: An Incremental Improvement [`[pdf]`](https://pjreddie.com/media/files/papers/YOLOv3) [`[code - pytorch]`](https://github.com/eriklindernoren/PyTorch-YOLOv3)

4. **[SSD]** SSD: Single Shot MultiBox Detector [`[pdf]`](https://arxiv.org/pdf/1512.02325) [`[official code - caffe]`](https://github.com/weiliu89/caffe/tree/ssd)[`[code - pytorch]`](https://github.com/amdegroot/ssd.pytorch)


5. **[RetinaNet]** Focal Loss for Dense Object Detection [`[pdf]`](https://arxiv.org/pdf/1708.02002) [`[offical code - pytorch]`](https://github.com/facebookresearch/Detectron)[`[code - pytorch]`](https://github.com/yhenon/pytorch-retinanet)

### NMS
1. **[Soft-NMS]** Improving Object Detection With One Line of Code [`[pdf]`](https://arxiv.org/pdf/1704.04503) [`[offical code - caffe]`](https://github.com/bharatsingh430/soft-nms)[`[code - pytorch]`](https://github.com/wushuang01/soft-nms/blob/master/soft-nms.py)

2. **[Softer-NMS]** Softer-NMS: Rethinking Bounding Box Regression for Accurate Object Detection [`[pdf]`](https://arxiv.org/pdf/1809.08545v1) [`[offical code - Caffe2]`](https://github.com/yihui-he/softer-NMS)

3. **[KL-Loss]** Bounding Box Regression with Uncertainty for Accurate Object Detection [`[pdf]`](https://arxiv.org/pdf/1809.08545) [`[offical code - Caffe2]`](https://github.com/yihui-he/KL-Loss)

4. Learning non-maximum suppression [`[pdf]`](https://arxiv.org/pdf/1705.02950)

5. **[Relation-Network]** Relation Networks for Object Detection [`[pdf]`](https://arxiv.org/pdf/1711.11575) [`[offical code - mxnet]`](https://github.com/msracver/Relation-Networks-for-Object-Detection)

6. **[PrPool]** Acquisition of Localization Conﬁdence for Accurate Object Detection [`[pdf]`](https://arxiv.org/pdf/1807.11590) [`[offical code - pytorch]`](https://github.com/vacancy/PreciseRoIPooling)

7. **[Adaptive-NMS]** Adaptive NMS: Reﬁning Pedestrian Detection in a Crowd [`[pdf]`](https://arxiv.org/pdf/1904.03629) 


### Anchor-free
1. **[CornerNet]** CornerNet: Detecting Objects as Paired Keypoints [`[pdf]`](https://arxiv.org/pdf/1808.01244) [`[offical code - pytorch]`](https://github.com/princeton-vl/CornerNet)

2. **[CornerNet-Lite]** CornerNet-Lite: Efficient Keypoint Based Object Detection [`[pdf]`](https://arxiv.org/pdf/1904.08900) [`[offical code - pytorch]`](https://github.com/princeton-vl/CornerNet-Lite)

3. **[FSAF]** Feature Selective Anchor-Free Module for Single-Shot Object Detection [`[pdf]`](https://arxiv.org/pdf/1903.00621) [`[code - tensorflow]`](https://github.com/xuannianz/FSAF)[`[code - pytorch]`](https://github.com/hdjang/Feature-Selective-Anchor-Free-Module-for-Single-Shot-Object-Detection)

4. **[ExtremeNet]** Bottom-up Object Detection by Grouping Extreme and Center Points [`[pdf]`](https://arxiv.org/pdf/1901.08043) [`[offical code - pytorch]`](https://github.com/xingyizhou/ExtremeNet)

5. **[FoveaBox]** FoveaBox: Beyond Anchor-based Object Detector [`[pdf]`](https://arxiv.org/pdf/1904.03797) [`[offical code - pytorch]`](https://github.com/taokong/FoveaBox)

6. **[FCOS]** FCOS: Fully Convolutional One-Stage Object Detection [`[pdf]`](https://arxiv.org/pdf/1904.01355) [`[ code - pytorch]`](https://github.com/yqyao/FCOS_PLUS)

7. **[CenterNet]** Objects as Points [`[pdf]`](https://arxiv.org/pdf/1904.07850) [`[offical code - pytorch]`](https://github.com/xingyizhou/CenterNet)

8. AFP-Net: Realtime Anchor-Free Polyp Detection in Colonoscopy [`[pdf]`](https://arxiv.org/pdf/1909.02477)

9. **[DenseBox]** Objects as Points [`[pdf]`](https://arxiv.org/pdf/1509.04874) [`[code - pytorch]`](https://github.com/CaptainEven/DenseBox)

10. **[RepPoints]** RepPoints: Point Set Representation for Object Detection [`[pdf]`](https://arxiv.org/pdf/1904.11490)[`[offical code - pytorch]`](https://github.com/microsoft/RepPoints)

## Instance Segmentation
1. **[Mask R-CNN]** Mask R-CNN [`[pdf]`](https://arxiv.org/pdf/1703.06870) [`[offical code - caffe]`](https://github.com/facebookresearch/Detectron)[`[code - pytorch]`](https://github.com/multimodallearning/pytorch-mask-rcnn)

2. **[SDS]** Simultaneous Detection and Segmentation [`[pdf]`](https://arxiv.org/abs/1407.1808)

3. **[InstanceCut]** InstanceCut: from Edges to Instances with MultiCut [`[pdf]`](https://arxiv.org/abs/1611.08272)[`[offical code]`](https://github.com/mrlooi/multicut_inference)

3. **[InstanceFCN]** Instance-sensitive Fully Convolutional Networks [`[pdf]`](https://arxiv.org/abs/1603.08678)

4. **[FCIS]** Fully Convolutional Instance-aware Semantic Segmentation [`[pdf]`](https://arxiv.org/abs/1611.07709) [`[offical code - mxnet]`](https://github.com/msracver/FCIS)

5. **[PolarMask]** PolarMask: Single Shot Instance Segmentation with Polar Representation [`[pdf]`](https://arxiv.org/abs/1909.13226) [`[offical code - pytorch]`](https://github.com/xieenze/PolarMask)

## Video Object Detection
1. **[FGFA]** Flow-Guided Feature Aggregation for Video Object Detection [`[pdf]`](https://arxiv.org/abs/1703.10025) [`[offical code - mxnet]`](https://github.com/msracver/Flow-Guided-Feature-Aggregation)

2. **[TCN]** Object Detection from Video Tubelets with Convolutional Neural Networks [`[pdf]`](https://arxiv.org/abs/1604.04053) [`[offical code - caffe]`](https://github.com/myfavouritekk/vdetlib)

3. **[Seq-NMS]** Seq-NMS for Video Object Detection [`[pdf]`](https://arxiv.org/abs/1602.08465)

## Others
1. **[Libra R-CNN]** Libra R-CNN: Balanced Learning for Object Detection [`[pdf]`](https://arxiv.org/pdf/1904.02701)[`[offical code - pytorch]`](https://github.com/OceanPang/Libra_R-CNN)

2. **[Cascade R-CNN]** Cascade R-CNN: Delving into High Quality Object Detection [`[pdf]`](https://arxiv.org/pdf/1712.00726) [`[offical code - caffe]`](https://github.com/zhaoweicai/cascade-rcnn)[`[code - pytorch]`](https://github.com/guoruoqian/cascade-rcnn_Pytorch)

3. **[FPN]** Feature Pyramid Networks for Object Detection [`[pdf]`](https://arxiv.org/pdf/1612.03144) [`[code - pytorch]`](https://github.com/jwyang/fpn.pytorch)

4. **[M2Det]** M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network [`[pdf]`](https://arxiv.org/pdf/1811.04533) [`[offical code - pytorch]`](https://github.com/qijiezhao/M2Det)

5. **[OHEM]** Training Region-based Object Detectors with Online Hard Example Mining [`[pdf]`](https://arxiv.org/pdf/1604.03540) [`[offical code - caffe]`](https://github.com/abhi2610/ohem)[`[code - pytorch]`](https://github.com/gurkirt/FPN.pytorch1.0)

6. **[GIoU]** Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression [`[pdf]`](https://arxiv.org/pdf/1902.09630) [`[code - pytorch]`](https://github.com/diggerdu/Generalized-Intersection-over-Union)[`[code - tensorflow]`](https://github.com/generalized-iou/g-tensorflow-models)

7. **[GA-RPN]** Region Proposal by Guided Anchoring [`[pdf]`](https://arxiv.org/pdf/1901.03278) [`[offical code - pytorch]`](https://github.com/open-mmlab/mmdetection)

8. Multi-Oriented Scene Text Detection via Corner Localization and Region Segmentation [`[pdf]`](https://arxiv.org/pdf/1802.08948) [`[offical code - pytorch]`](https://github.com/lvpengyuan/corner)

9. Precise Detection in Densely Packed Scenes [`[pdf]`](https://arxiv.org/abs/1904.00853) [`[offical code - tensorflow]`](http://www.github.com/eg4000/SKU110K_CVPR19)

10. ScratchDet: Training Single-Shot Object Detectors from Scratch [`[pdf]`](https://arxiv.org/abs/1810.08425) [`[offical code - caffe]`](https://github.com/KimSoybean/ScratchDet)

11. **[ELASTIC]** ELASTIC: Improving CNNs with Dynamic Scaling Policies [`[pdf]`](https://arxiv.org/abs/1812.05262) [`[offical code - pytorch]`](https://github.com/allenai/elastic)

12. **[FractalNet]** FractalNet: Ultra-Deep Neural Networks without Residuals [`[pdf]`](https://arxiv.org/abs/1605.07648) [`[code - pytorch]`](https://github.com/khanrc/pt.fractalnet)[`[code - tensorflow]`](https://github.com/snf/keras-fractalnet)

13. Attention Is All You Need [`[pdf]`](https://arxiv.org/abs/1706.03762)

14. **[GHM]** Gradient Harmonized Single-stage Detector [`[pdf]`](https://arxiv.org/abs/1811.05181) [`[offical code - pytorch]`](https://github.com/libuyu/GHM_Detection)


# Acknowledgements
- Significant amounts of content are based on the [deep learning object detection](https://github.com/hoya012/deep_learning_object_detection)