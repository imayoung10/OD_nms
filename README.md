# OD_nms
## Object Detection에서 모든 모델에 적용 가능한 Postprocess function을 작성.
### Method
대부분의 모델 형태 : (B, A*(C+4), H, W)
B : batch, A: anchors (없는 경우 1), C: 클래스 수

1. (B, D, H*W) 형태로 변환
2. object logit이 있는 경우와 없는 경우로 나눔\
    2.1 있는 경우\
        class probability에 곱해서 최종 prediction 생성\
    2.2 없는 경우\
        BBox와 class score 분리\
3. score로 필터링
4. nms 적용