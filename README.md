# YOLOv4_pytorch

### TODO List

- [x] Dataset
- [ ] Model
- [ ] Loss
- [X] Coder
- [x] burn-in 하기
- [x] scheduler 확립하기
- [x] IT
- [ ] M(Mosaic augmentation)
- [ ] OA(Optimized Anchors)
 
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
 
### Experiment

268

265

- [x] Model 추가

|methods                   | Traning Dataset        |    Testing Dataset     | Resolution | AP      |AP50   |AP75    | Time | Fps  |
|--------------------------|------------------------| ---------------------- | ---------- | ------- |-------|--------|:----:| ---- |
|papers(YOLOv3)            | COCOtrain2017          |  COCO test-dev         | 416 x 416  |  31.0   |55.3   |34.4    |29    |34.48 |
|papers                    | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |  -      |-      |-       |-     |-     |
|yolov3 + CSP              | COCOtrain2017          |  COCO test-dev         | 416 x 416  |- |-   |-|-|- |
|yolov3 + CSP              | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |**38.0** |59.9  |40.8   |||
|yolov3 + CSP + giou loss  | COCOtrain2017          |  COCO test-dev         | 416 x 416  |? |? |?  |||
|yolov3 + CSP + giou loss  | COCOtrain2017          |  COCOval2017(minival)  | 416 x 416  |39.xx|? |?   |||
|YOLOv4                    | COCOtrain2017          |  COCO test-dev         | 416 x 416  |0.412|0.628|0.448|||
|YOLOv4                    | COCOtrain2017          |  COCO test-dev         | 512 x 512  |0.430|0.649|0.465|||


- GIOU loss 서버실 확장 때문에 한번 끊었다가 감. 155 부터 시작하면 될듯 (2021/05/24 해결)
![](./figure/giou_155_epochs.JPG)


- experiments1

    yolov3 + CSP 
    
    ```
    lr = 1e-3
    epoch = 273 
    burn_in = 4000
    batch_size = 64
    optimizer = SGD
    lr decay = step LR [218, 246]
    best_epch = 265
    ```

    |experiments    | Dataset | Resolution |  base detector           | AP     |AP50   |AP75   |
    |---------------|---------| ---------- | ------------------------ | ------ |-------|-------|
    |exp1           | minival | 416 x 416  | yolov3 + CSP             |**38.0**|59.9   |40.8   |

- experiments2 (~21.05.24)

    yolov3 + CSP + GIoULoss
    
    ```
    lr = 1e-3
    epoch = 273 
    burn_in = 4000
    batch_size = 64
    optimizer = SGD
    lr decay = step LR [218, 246]
    ```

    |experiments    | Dataset | Resolution |  base detector           | AP     |AP50   |AP75   |
    |---------------|---------| ---------- | ------------------------ | ------ |-------|-------|
    |exp2           | minival | 416 x 416  | yolov3 + CSP + GIoU      |**39.0**|??   |??   |
   

- experiments3

    yolov3 + CSP + GIoULoss + IT(Iou threshold) + M(Mosaic) + OA(Optimized Anchor) 
    
    ```
    lr = 1e-3
    epoch = 273 
    burn_in = 4000
    batch_size = 64
    optimizer = SGD
    lr decay = cosine annealing lr scheduler
    ```

    |experiments    | Dataset | Resolution |  base detector                         | AP     |AP50   |AP75   |
    |---------------|---------| ---------- | -------------------------------------- | ------ |-------|-------|
    |exp3           | minival | 512 x 512  | yolov3 + CSP + GIoU + IT + M + OA      |-|-|-|
    |YOLOv4         | COCO test-dev | 512 x 512 | YOLOv4                            |0.430   |0.649  |0.465  |
