# YOLOv4_pytorch

### TODO List

- [x] Dataset
- [ ] Model
- [ ] Loss
- [X] Coder
- [x] burn-in 하기
- [x] scheduler 확립하기
 
### Experiment

- [x] Model 추가

|methods                   | Traning Dataset        |    Testing Dataset     | Resolution | AP      |AP50   |AP75    | Time | Fps  |
|--------------------------|------------------------| ---------------------- | ---------- | ------- |-------|--------|:----:| ---- |
|papers                    | COCOtrain2017          |  COCO test-dev         | 416 x 416  |  31.0   |55.3   |34.4    |29    |34.48 |
|papers                    | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |  -      |-      |-       |-     |-     |
|yolov3 + CSP              | COCOtrain2017          |  COCO test-dev         | 416 x 416  |- |-   |-|-|- |
|yolov3 + CSP              | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |**38.0** |59.9  |40.8   |||
|yolov3 + CSP + giou loss  | COCOtrain2017          |  COCO test-dev         | 416 x 416  |? |? |?  |||
|yolov3 + CSP + giou loss  | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |? |? |?   |||


268 epoch 기준

### scheduler

- we use step LR scheduler and learning rate warmup(burning) scheme 

1833 x 218 = 399594

1833 x 246 = 450918

1833 x 273 = 500409

### burn-in

- batch 64 & iteration 1000

- lr : 0 to 1e-3 (i/1000)

### trining

- batch : 64
- scheduler : step LR
- loss : sse + bce
- dataset : coco
- epoch : 273
- gpu : nvidia geforce rtx 3090 * 2EA
