import torch
import torch.nn.functional as F
import torchvision

def postprocess(output, C, score_thr, iou_thr):
    if len(output.shape) > 3:
        output.reshape(output.shape[0], output.shape[1], -1)
    
    output.permute(0, 2, 1)
    
    D = output.shape[-1]
    if D == 4 + C + 1:
        has_obj = True
    elif D == 4 + C:
        has_obj = False
    elif D == C:
        has_obj = False
    else:
        raise ValueError("Unsupported output dim")

    ## obj_logit 있는/없는 경우
    if has_obj:
        box    = output[..., :4]
        obj    = output[..., 4].sigmoid()
        cls_logit = output[..., 5:]
        cls_prob  = F.softmax(cls_logit, dim=-1)
        score, label = (obj.unsqueeze(-1) * cls_prob).max(-1)
    else:
        if D == 4 + C:
            box      = output[..., :4]
            cls_logit = output[..., 4:] 
            cls_prob  = F.softmax(cls_logit, dim=-1)
            score, label = cls_prob[..., :-1].max(-1)
        else:
            # bbox는 별도 tensor (B, M, 4) 로 이미 준비되어 있다고 가정
            cls_prob = output.sigmoid() 
            score, label = cls_prob.max(-1)
    
    keep = score > score_thr
    boxes = box[keep]
    scores = score[keep]
    labels = label[keep]
    final_idx = torchvision.ops.nms(boxes, scores, iou_thr)
    
    return boxes[final_idx], scores[final_idx], labels[final_idx]