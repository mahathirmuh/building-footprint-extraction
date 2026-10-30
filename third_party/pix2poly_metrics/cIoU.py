# Vendored from Pix2Poly (https://github.com/yeshwanth95/Pix2Poly), commit 4e4c00359b41f31fe80f7fbdab0fa50eb7a46ab8,
# eval/hisup_eval_utils/metrics/cIoU.py (MIT, third_party/licenses/Pix2Poly-LICENSE.txt). The verbatim copy is
# third_party/pix2poly/cIoU.py. Wrapper changes only (docs/EVALUATION.md §2.3), each marked "[wrapper N]"; the
# per-image formula, the image set (only images that have predictions), and the pooling are unchanged:
#   1. compute_IoU_cIoU() returns the means it prints, plus the values of every image (iou, ciou, N, N_GT).
#   2. img_ids: optional subset of image ids, iterated in the original order.
#   3. mask_gti is reset for every image. The original keeps the previous image's mask when an image has no GT
#      annotation (or raises UnboundLocalError when that image comes first); N_GT was already reset there.
#   4. progress=False skips the running-mean progress text, which recomputes four means over all images seen so far
#      on every image (quadratic in the number of images).
#   5. A prediction pycocotools cannot read re-raises its error instead of opening the ipdb debugger.
#   6. A mean over an empty list is returned as None (the original prints nan).
"""
This is the code from https://github.com/zorzi-s/PolyWorldPretrainedNetwork.
@article{zorzi2021polyworld,
  title={PolyWorld: Polygonal Building Extraction with Graph Neural Networks in Satellite Images},
  author={Zorzi, Stefano and Bazrafkan, Shabab and Habenschuss, Stefan and Fraundorfer, Friedrich},
  journal={arXiv preprint arXiv:2111.15491},
  year={2021}
}
"""

from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm

def calc_IoU(a, b):
    i = np.logical_and(a, b)
    u = np.logical_or(a, b)
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou

def _mean_or_none(values):  # [wrapper 6]
    return float(np.mean(values)) if len(values) else None

def compute_IoU_cIoU(input_json, gti_annotations, img_ids=None, progress=True):
    # Ground truth annotations
    coco_gti = COCO(gti_annotations)

    # Predictions annotations
    submission_file = json.loads(open(input_json).read())
    coco = COCO(gti_annotations)
    coco = coco.loadRes(submission_file)


    image_ids = coco.getImgIds(catIds=coco.getCatIds())
    if img_ids is not None:  # [wrapper 2]
        keep = set(img_ids)
        image_ids = [image_id for image_id in image_ids if image_id in keep]
    bar = tqdm(image_ids, disable=not progress)

    list_iou = []
    list_ciou = []
    pss = []
    rel_difs = []
    n_ratios = []
    per_image = {}  # [wrapper 1]
    for image_id in bar:

        img = coco.loadImgs(image_id)[0]

        annotation_ids = coco.getAnnIds(imgIds=img['id'])
        annotations = coco.loadAnns(annotation_ids)
        N = 0
        for _idx, annotation in enumerate(annotations):
            try:
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
            except Exception:
                raise  # [wrapper 5] was: import ipdb; ipdb.set_trace()
            m = cocomask.decode(rle)
            if _idx == 0:
                mask = m.reshape((img['height'], img['width']))
                N = len(annotation['segmentation'][0]) // 2
            else:
                mask = mask + m.reshape((img['height'], img['width']))
                N = N + len(annotation['segmentation'][0]) // 2

        mask = mask != 0


        annotation_ids = coco_gti.getAnnIds(imgIds=img['id'])
        annotations = coco_gti.loadAnns(annotation_ids)
        N_GT = 0
        mask_gti = np.zeros((img['height'], img['width']))  # [wrapper 3]
        for _idx, annotation in enumerate(annotations):
            if any(annotation['segmentation']):
                rle = cocomask.frPyObjects(annotation['segmentation'], img['height'], img['width'])
                m = cocomask.decode(rle)
            else:
                annotation['segmentation'] = [[]]
                m = np.zeros((img['height'], img['width']))
            if m.ndim > 2:
                m = np.clip(0, 1, m.sum(axis=-1))
            if _idx == 0:
                mask_gti = m.reshape((img['height'], img['width']))
                N_GT = len(annotation['segmentation'][0]) // 2
            else:
                mask_gti = mask_gti + m.reshape((img['height'], img['width']))
                N_GT = N_GT + len(annotation['segmentation'][0]) // 2

        mask_gti = mask_gti != 0

        ps = 1 - np.abs(N - N_GT) / (N + N_GT + 1e-9)
        rel_dif = np.abs(N - N_GT) / (N + N_GT + 1e-9)
        iou = calc_IoU(mask, mask_gti)
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        pss.append(ps)
        rel_difs.append(rel_dif)
        if N_GT > 0:
            nr = N / N_GT
            n_ratios.append(nr)
        per_image[image_id] = {"iou": float(iou), "ciou": float(iou * ps), "n": N, "n_gt": N_GT}  # [wrapper 1]

        if progress:  # [wrapper 4]
            bar.set_description("iou: %2.4f, c-iou: %2.4f, ps:%2.4f, rd:%2.4f" % (np.mean(list_iou), np.mean(list_ciou), np.mean(pss), np.mean(rel_difs)))
            bar.refresh()

    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
    print("Mean N-Relative Difference: ", np.mean(rel_difs))
    print("Mean N-Ratio: ", np.mean(n_ratios))

    return {  # [wrapper 1]
        "iou": _mean_or_none(list_iou),
        "ciou": _mean_or_none(list_ciou),
        "n_rel_diff": _mean_or_none(rel_difs),
        "n_ratio": _mean_or_none(n_ratios),
        "n_images": len(list_iou),
        "per_image": per_image,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    args = parser.parse_args()

    gt_file = args.gt_file
    dt_file = args.dt_file
    compute_IoU_cIoU(input_json=dt_file,
                    gti_annotations=gt_file)
