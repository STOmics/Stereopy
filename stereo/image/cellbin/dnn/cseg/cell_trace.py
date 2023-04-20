import cv2


def get_trace(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    h, w = mask.shape[: 2]
    output = []
    for i in range(num_labels):
        box_w, box_h, area = stats[i][2:]
        if box_h == h and box_w == w:
            continue
        output.append([box_h, box_w, area])
    return output
