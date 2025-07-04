import numpy as np
import cv2
from tqdm import tqdm


SILHOUETTE_IDS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
]
LIP_IDS = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
           146, 91, 181, 84, 17, 314, 405, 321, 375]


def get_face_mask(landmarks, height, width, out_path=None, expand_ratio=1.2):
    """
    Generate a face mask based on the given landmarks.

    Args:
        landmarks (numpy.ndarray): The landmarks of the face.
        height (int): The height of the output face mask image.
        width (int): The width of the output face mask image.
        out_path (pathlib.Path): The path to save the face mask image.
        expand_ratio (float): Expand ratio of mask.
    Returns:
        None. The face mask image is saved at the specified path.
    """
    face_landmarks = np.take(landmarks, SILHOUETTE_IDS, 0)
    min_xy_face = np.round(np.min(face_landmarks, 0))
    max_xy_face = np.round(np.max(face_landmarks, 0))
    min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1] = expand_region(
        [min_xy_face[0], max_xy_face[0], min_xy_face[1], max_xy_face[1]], width, height, expand_ratio)
    face_mask = np.zeros((height, width), dtype=np.uint8)
    face_mask[round(min_xy_face[1]):round(max_xy_face[1]),
              round(min_xy_face[0]):round(max_xy_face[0])] = 255
    if out_path:
        cv2.imwrite(str(out_path), face_mask)
        return None
    
    bbox = [min_xy_face[0], min_xy_face[1], max_xy_face[0], max_xy_face[1]]
    return face_mask, bbox 


def expand_region(region, image_w, image_h, expand_ratio=1.0):
    """
    Expand the given region by a specified ratio.
    Args:
        region (tuple): A tuple containing the coordinates (min_x, max_x, min_y, max_y) of the region.
        image_w (int): The width of the image.
        image_h (int): The height of the image.
        expand_ratio (float, optional): The ratio by which the region should be expanded. Defaults to 1.0.

    Returns:
        tuple: A tuple containing the expanded coordinates (min_x, max_x, min_y, max_y) of the region.
    """

    min_x, max_x, min_y, max_y = region
    mid_x = (max_x + min_x) // 2
    side_len_x = (max_x - min_x) * expand_ratio
    mid_y = (max_y + min_y) // 2
    side_len_y = (max_y - min_y) * expand_ratio
    min_x = mid_x - side_len_x // 2
    max_x = mid_x + side_len_x // 2
    min_y = mid_y - side_len_y // 2
    max_y = mid_y + side_len_y // 2
    

    if min_x < 0:
        #max_x -= min_x
        min_x = 0
    if max_x > image_w:
        #min_x -= max_x - image_w
        max_x = image_w
    if min_y < 0:
        #max_y -= min_y
        min_y = 0
    if max_y > image_h:
        #min_y -= max_y - image_h
        max_y = image_h

    return round(min_x), round(max_x), round(min_y), round(max_y)


def get_union_mask(masks):
    """
    Compute the union of a list of masks.

    This function takes a list of masks and computes their union by taking the maximum value at each pixel location.
    Additionally, it finds the bounding box of the non-zero regions in the mask and sets the bounding box area to white.

    Args:
        masks (list of np.ndarray): List of masks to be combined.

    Returns:
        np.ndarray: The union of the input masks.
    """
    union_mask = None
    for mask in masks:
        if union_mask is None:
            union_mask = mask
        else:
            union_mask = np.maximum(union_mask, mask)

    if union_mask is not None:
        # Find the bounding box of the non-zero regions in the mask
        rows = np.any(union_mask, axis=1)
        cols = np.any(union_mask, axis=0)
        try:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
        except Exception as e:
            print(str(e))
            return 0.0

        # Set bounding box area to white
        union_mask[ymin: ymax + 1, xmin: xmax + 1] = np.max(union_mask)

    return union_mask


def get_union_bbox(bboxes):
    min_x = np.min(bboxes[:, 0])
    min_y = np.min(bboxes[:, 1])
    max_x = np.max(bboxes[:, 2])
    max_y = np.max(bboxes[:, 3])
    return min_x, min_y, max_x, max_y

def mask2bbox(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return xmin, ymin, xmax, ymax


def form_batches(frame_numbers, batch_size=9, stride=5): 
    batches = []
    n = 0 
    for i, _ in enumerate(frame_numbers): 
        if i<n: 
            continue 
        start = i 
        end = start + batch_size
        sf = frame_numbers[start:end]
        if sf[-1] - sf[0] != batch_size - 1: 
            n = i+1 
            continue 
        else:
            batches.append(sf)
            n = i+stride
    batchs = np.stack(batches)
    batches = batchs[:, 2:7]
    return np.asarray(batches), (batches[:, -1] - batches[:, 0] == stride-1).all()


def is_bbox_within_image(bbox, height, width):
    min_x, min_y, max_x, max_y = bbox
    return min_x >= 0 and min_y >= 0 and max_x <= width and max_y <= height

def make_square_bbox(bbox, height, width):
    min_x, min_y, max_x, max_y = bbox
    size = min(max_x-min_x, max_y-min_y)
    ctr_x = (max_x + min_x) // 2
    ctr_y = (max_y + min_y) // 2
    radius = size // 2
    out = [ctr_x-radius, ctr_y-radius, ctr_x+radius, ctr_y+radius]
    return is_bbox_within_image(out, height, width), out


def get_face_mask_batch(face_landmarks, hw, frame_numbers, batch_size=9, stride=5, expand_ratio=2.3):
    h, w = hw[0] #videos have same height and width 
    batches, valid = form_batches(frame_numbers, batch_size=batch_size, stride=stride)
    if not valid: 
        print("no valid batches found")
        return None, None 
    bboxes = []
    for n, batch in tqdm(enumerate(batches)): 
        face_masks = []
        for frame_number in batch: 
            index = frame_numbers.tolist().index(frame_number)
            face_mask, _ = get_face_mask(face_landmarks[index], h, w, expand_ratio=expand_ratio)
            face_masks.append(face_mask)
        
        # get face attributes 
        face_mask = get_union_mask(face_masks)
        face_valid, bbox = make_square_bbox(mask2bbox(face_mask), h, w)
        if not face_valid:
            bboxes.append([0, 0, 0, 0]+[n]+[-1])
            continue 
        bboxes.append(bbox+[n]+[1])
    return batches, np.asarray(bboxes)

if __name__ == "__main__":
    pass 