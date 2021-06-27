import os
import sys
import torch
import random
import torchvision.transforms.functional as FT
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import find_jaccard_overlap
from config import device

# expand
# random crop
# flip
# photometric distort


def expand(image, boxes, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    new_image = torch.ones((3, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes


def random_crop(image, boxes, labels):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels


def flip(image, boxes):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = FT.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0]
    new_boxes[:, 2] = image.width - boxes[:, 2]
    new_boxes = new_boxes[:, [2, 1, 0, 3]]

    return new_image, new_boxes


def photometric_distort(image):
    """
    Distort brightness, contrast, saturation, and hue, each with a 50% chance, in random order.

    :param image: image, a PIL Image
    :return: distorted image
    """
    new_image = image

    distortions = [FT.adjust_brightness,
                   FT.adjust_contrast,
                   FT.adjust_saturation,
                   FT.adjust_hue]

    random.shuffle(distortions)

    for d in distortions:
        if random.random() < 0.5:
            if d.__name__ is 'adjust_hue':
                # Caffe repo uses a 'hue_delta' of 18 - we divide by 255 because PyTorch needs a normalized value
                adjust_factor = random.uniform(-18 / 255., 18 / 255.)
            else:
                # Caffe repo uses 'lower' and 'upper' values of 0.5 and 1.5 for brightness, contrast, and saturation
                adjust_factor = random.uniform(0.5, 1.5)

            # Apply this distortion
            new_image = d(new_image, adjust_factor)

    return new_image


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, boxes, labels, split, transform_list, new_size, zero_to_one_coord=True):

    allowed_tf_list = ['photo', 'expand', 'crop', 'flip', 'resize']
    assert split in {'train', 'test'}
    for tf in transform_list:
        assert tf in allowed_tf_list

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'train':

        if 'photo' in transform_list:
            # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
            new_image = photometric_distort(new_image)

        new_image = FT.to_tensor(new_image)

        if 'expand' in transform_list:
            each_img_mean = torch.mean(new_image, (1, 2))
            # Expand image (zoom out)
            if random.random() < 0.5:
                new_image, new_boxes = expand(new_image, boxes, filler=each_img_mean)

        if 'crop' in transform_list:
            new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        new_image = FT.to_pil_image(new_image)

        if 'flip' in transform_list:
            # Flip image with a 50% chance
            if random.random() < 0.5:
                new_image, new_boxes = flip(new_image, new_boxes)

    if 'resize' in transform_list:
        # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
        new_image, new_boxes = resize(new_image, new_boxes, (new_size, new_size), zero_to_one_coord)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels


def mosaic(images, boxes, labels, size):
    new_image_0, new_image_1, new_image_2, new_image_3 = images
    new_boxes_0, new_boxes_1, new_boxes_2, new_boxes_3 = boxes
    new_labels_0, new_labels_1, new_labels_2, new_labels_3 = labels

    # Image to tensor
    new_image_0 = FT.to_tensor(new_image_0)
    new_image_1 = FT.to_tensor(new_image_1)
    new_image_2 = FT.to_tensor(new_image_2)
    new_image_3 = FT.to_tensor(new_image_3)

    new_image_0, new_boxes_0, new_labels_0 = random_crop(new_image_0, new_boxes_0, new_labels_0)
    new_image_1, new_boxes_1, new_labels_1 = random_crop(new_image_1, new_boxes_1, new_labels_1)
    new_image_2, new_boxes_2, new_labels_2 = random_crop(new_image_2, new_boxes_2, new_labels_2)
    new_image_3, new_boxes_3, new_labels_3 = random_crop(new_image_3, new_boxes_3, new_labels_3)

    # tensor to Image
    new_image_0 = FT.to_pil_image(new_image_0)
    new_image_1 = FT.to_pil_image(new_image_1)
    new_image_2 = FT.to_pil_image(new_image_2)
    new_image_3 = FT.to_pil_image(new_image_3)

    new_size = size//2
    new_image_0, new_boxes_0 = resize(new_image_0, new_boxes_0, (new_size, new_size), False)
    new_image_1, new_boxes_1 = resize(new_image_1, new_boxes_1, (new_size, new_size), False)
    new_image_2, new_boxes_2 = resize(new_image_2, new_boxes_2, (new_size, new_size), False)
    new_image_3, new_boxes_3 = resize(new_image_3, new_boxes_3, (new_size, new_size), False)

    # bbox 바꾸는 부분
    new_boxes_1[:, 0] = new_boxes_1[:, 0] + new_size
    new_boxes_1[:, 2] = new_boxes_1[:, 2] + new_size

    new_boxes_2[:, 1] = new_boxes_2[:, 1] + new_size
    new_boxes_2[:, 3] = new_boxes_2[:, 3] + new_size

    new_boxes_3[:, 0] = new_boxes_3[:, 0] + new_size
    new_boxes_3[:, 1] = new_boxes_3[:, 1] + new_size
    new_boxes_3[:, 2] = new_boxes_3[:, 2] + new_size
    new_boxes_3[:, 3] = new_boxes_3[:, 3] + new_size

    # 합치는 부분 - imgae - pil level 에서 합침 refer to https://note.nkmk.me/en/python-pillow-concat-images/
    new_image_01 = get_concat_h_cut_center(new_image_0, new_image_1)
    new_image_23 = get_concat_h_cut_center(new_image_2, new_image_3)
    new_image = get_concat_v_cut_center(new_image_01, new_image_23)
    new_boxes = torch.cat([new_boxes_0, new_boxes_1, new_boxes_2, new_boxes_3], dim=0)
    new_labels = torch.cat([new_labels_0, new_labels_1, new_labels_2, new_labels_3], dim=0)

    return new_image, new_boxes, new_labels


def get_concat_h_cut_center(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, min(im1.height, im2.height)))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, (im1.height - im2.height) // 2))
    return dst


def get_concat_v_cut_center(im1, im2):
    dst = Image.new('RGB', (min(im1.width, im2.width), im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, ((im1.width - im2.width) // 2, im1.height))
    return dst


def transform_mosaic(image, boxes, labels, split, transform_list, new_size,
                     len_of_dataset,
                     parser,
                     root=None,       # coco
                     coco=None,       # coco
                     set_name=None,   # coco
                     img_list=None,   # voc
                     anno_list=None,  # voc
                     zero_to_one_coord=True):

    allowed_tf_list = ['photo', 'expand', 'crop', 'flip', 'resize', 'mosaic']
    assert split in {'train', 'test'}
    for tf in transform_list:
        assert tf in allowed_tf_list

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if 'mosaic' in transform_list:
        # 1. index 구하기
        idx_mosaic_1 = random.randint(0, len_of_dataset - 1)
        idx_mosaic_2 = random.randint(0, len_of_dataset - 1)
        idx_mosaic_3 = random.randint(0, len_of_dataset - 1)

        # 2. image open 하기
        # coco
        if coco is not None:
            img_id_list = list(coco.imgToAnns.keys())

            img_id_1 = img_id_list[idx_mosaic_1]
            file_name_1 = coco.loadImgs(ids=img_id_1)[0]['file_name']
            file_path_1 = os.path.join(root, 'images', set_name, file_name_1)

            img_id_2 = img_id_list[idx_mosaic_2]
            file_name_2 = coco.loadImgs(ids=img_id_2)[0]['file_name']
            file_path_2 = os.path.join(root, 'images', set_name, file_name_2)

            img_id_3 = img_id_list[idx_mosaic_3]
            file_name_3 = coco.loadImgs(ids=img_id_3)[0]['file_name']
            file_path_3 = os.path.join(root, 'images', set_name, file_name_3)

            # make anno
            anno_ids_1 = coco.getAnnIds(imgIds=img_id_1)  # img id 에 해당하는 anno id 를 가져온다.
            anno_1 = coco.loadAnns(ids=anno_ids_1)  # anno id 에 해당하는 annotation 을 가져온다.

            anno_ids_2 = coco.getAnnIds(imgIds=img_id_2)  # img id 에 해당하는 anno id 를 가져온다.
            anno_2 = coco.loadAnns(ids=anno_ids_2)  # anno id 에 해당하는 annotation 을 가져온다.

            anno_ids_3 = coco.getAnnIds(imgIds=img_id_3)  # img id 에 해당하는 anno id 를 가져온다.
            anno_3 = coco.loadAnns(ids=anno_ids_3)  # anno id 에 해당하는 annotation 을 가져온다.

        # voc
        else:
            file_path_1 = img_list[idx_mosaic_1]
            file_path_2 = img_list[idx_mosaic_2]
            file_path_3 = img_list[idx_mosaic_3]

        new_image_1 = Image.open(file_path_1).convert('RGB')
        new_image_2 = Image.open(file_path_2).convert('RGB')
        new_image_3 = Image.open(file_path_3).convert('RGB')

        # 3. parsing 하기
        # for voc
        if anno_list is not None:
            new_boxes_1, new_labels_1 = parser(anno_list[idx_mosaic_1])
            new_boxes_1 = torch.FloatTensor(new_boxes_1)
            new_labels_1 = torch.LongTensor(new_labels_1)  # 0 ~ 19

            new_boxes_2, new_labels_2 = parser(anno_list[idx_mosaic_2])
            new_boxes_2 = torch.FloatTensor(new_boxes_2)
            new_labels_2 = torch.LongTensor(new_labels_2)  # 0 ~ 19

            new_boxes_3, new_labels_3 = parser(anno_list[idx_mosaic_3])
            new_boxes_3 = torch.FloatTensor(new_boxes_3)
            new_labels_3 = torch.LongTensor(new_labels_3)  # 0 ~ 19

        # for coco
        else:
            det_anno_1 = parser(anno_1)
            new_boxes_1 = torch.FloatTensor(det_anno_1[:, :4])  # numpy to Tensor
            new_labels_1 = torch.LongTensor(det_anno_1[:, 4])

            det_anno_2 = parser(anno_2)
            new_boxes_2 = torch.FloatTensor(det_anno_2[:, :4])  # numpy to Tensor
            new_labels_2 = torch.LongTensor(det_anno_2[:, 4])

            det_anno_3 = parser(anno_3)
            new_boxes_3 = torch.FloatTensor(det_anno_3[:, :4])  # numpy to Tensor
            new_labels_3 = torch.LongTensor(det_anno_3[:, 4])

    new_image = image
    new_boxes = boxes
    new_labels = labels

    # Skip the following operations for evaluation/testing
    if split == 'train':

        if 'mosaic' in transform_list:
            if random.random() > 0.5:
                images_ = (new_image, new_image_1, new_image_2, new_image_3)
                boxes_ = (new_boxes, new_boxes_1, new_boxes_2, new_boxes_3)
                labels_ = (new_labels, new_labels_1, new_labels_2, new_labels_3)
                new_image, new_boxes, new_labels = mosaic(images_, boxes_, labels_, new_size)

        if 'photo' in transform_list:
            # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
            new_image = photometric_distort(new_image)

        new_image = FT.to_tensor(new_image)

        if 'expand' in transform_list:
            each_img_mean = torch.mean(new_image, (1, 2))
            # Expand image (zoom out)
            if random.random() < 0.5:
                new_image, new_boxes = expand(new_image, new_boxes, filler=each_img_mean)

        if 'crop' in transform_list:
            new_image, new_boxes, new_labels = random_crop(new_image, new_boxes, new_labels)

        new_image = FT.to_pil_image(new_image)

        if 'flip' in transform_list:
            # Flip image with a 50% chance
            if random.random() < 0.5:
                new_image, new_boxes = flip(new_image, new_boxes)

    if 'resize' in transform_list:
        # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form
        new_image, new_boxes = resize(new_image, new_boxes, (new_size, new_size), zero_to_one_coord)

    # Convert PIL image to Torch tensor
    new_image = FT.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    new_image = FT.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels

