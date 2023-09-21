import os
import random
from typing import Optional, List, Tuple

import cv2
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist

from itertools import count, filterfalse

Box = Tuple[float, float, float, float]


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    """ From yolov7 utils.plots """
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_skeleton_kpts(im, kpts, steps):
    """ From yolov7 utils.plots """
    # Plot the skeleton and keypointsfor coco datatset
    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

    pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
    pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
    radius = 5
    num_kpts = len(kpts) // steps

    for kid in range(num_kpts):
        r, g, b = pose_kpt_color[kid]
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        if not (x_coord % 640 == 0 or y_coord % 640 == 0):
            if steps == 3:
                conf = kpts[steps * kid + 2]
                if conf < 0.5:
                    continue
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)


def get_first_new_id(id_list: List[int]) -> int:
    """ Returns the first id not present in the dataset """
    return next(filterfalse(set(id_list).__contains__, count(1)))


def plot_segmentation(points: List[Tuple[int, int]], img: np.ndarray, color: Optional[Tuple[int, int, int]] = None):
    """ Plots the segmentation (list of points) in the image """
    # Plots one bounding box on image img
    color = color or [random.randint(0, 255) for _ in range(3)]
    polygons = [np.reshape(p, (-1, 2)).astype(np.int32) for p in points]
    cv2.fillPoly(img, pts=polygons, color=color)  # filled


def merge_or_append(list_dic: List[dict], new_dic: dict, key='id'):
    """
    If a dictionary in the list has the same value associated with the key as the new dictionary,
    merge them, otherwise add the new dictionary to the end of the list.
    """
    list_dic = list_dic.copy()
    update = False
    for i, dic in enumerate(list_dic):
        if dic[key] == new_dic[key]:
            list_dic[i] = dic | new_dic
            update = True
            break
    if not update:
        list_dic.append(new_dic)
    return list_dic


def read_json(json_file: str):
    with open(json_file, 'r') as f:
        d = json.load(f)
    return d


def bbox_coco_to_yolo(bbox: Box, width: int, height: int) -> Box:
    """ [top_left_x, top_left_y, pixel_width, pixel_height] -> [center_x, center_y, prop_width, prop_height] """
    center_x = (bbox[0] + bbox[2] / 2) / width
    x_size = bbox[2] / width
    center_y = (bbox[1] + bbox[3] / 2) / height
    y_size = bbox[3] / height
    return center_x, center_y, x_size, y_size


def bbox_yolo_to_coco(bbox: Box, width: int, height: int) -> Box:
    """ [center_x, center_y, prop_width, prop_height] -> [top_left_x, top_left_y, pixel_width, pixel_height] """
    x = (bbox[0] - bbox[2] / 2) * width
    x_size = bbox[2] * width
    y = (bbox[1] - bbox[3] / 2) * height
    y_size = bbox[3] * height
    return x, y, x_size, y_size


def bbox_coco_area_ratio(bbox_in: Box, bbox_out: Box) -> float:
    """ Intersection(bbox_in, bbox_out) / bbox_in """
    in_area = bbox_in[2] * bbox_in[3]
    x_size = min(bbox_in[0] + bbox_in[2], bbox_out[0] + bbox_out[2]) - max(bbox_in[0], bbox_out[0])
    y_size = min(bbox_in[1] + bbox_in[3], bbox_out[1] + bbox_out[3]) - max(bbox_in[1], bbox_out[1])
    if y_size < 0 or x_size < 0:
        return 0
    return x_size * y_size / in_area


def segmentation_radius(seg):
    """ Returns the maximum distance between two points of the segmentation. """
    # https://stackoverflow.com/questions/31667070/max-distance-between-2-points-in-a-data-set-and-identifying-the-points
    seg = np.array(seg).reshape((-1, 2))
    hull = ConvexHull(seg)
    hull_points = seg[hull.vertices, :]
    hdist = cdist(hull_points, hull_points, metric='euclidean')
    # bestpair = np.unravel_index(hdist.argmax(), hdist.shape)
    # print([hullpoints[bestpair[0]],hullpoints[bestpair[1]]])
    return np.max(hdist)


def distance_between_segmentations(segmentations: np.array) -> float:
    """ Returns the greater distance between two segmentations in the set of segmentations."""
    segmentations = [np.array(seg).reshape((-1, 2)) for seg in segmentations]
    hulls = [seg[ConvexHull(seg).vertices, :] for seg in segmentations]

    return np.max([np.min(cdist(h1, h2, metric='euclidean')) for h1 in hulls for h2 in hulls])


class NpEncoder(json.JSONEncoder):
    """ https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class CocoSet:
    """ CocoSet lets you easily manipulate Coco annotations, transforming them and exporting them in a format
    supported by YoloV7.

    For more information about the coco data format:
    https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch
    """

    def __init__(self, image_path: str):
        """ Init

        :param image_path: Directory containing the images associated with the coco dataset.
        """
        self.image_path = os.path.normpath(image_path)
        self.dic: dict = {}
        self.params: dict = {}
        self._available_ids: list = []
        self.ids: set = set()
        self.colors: dict = {}

    def load_coco(self, files: List[str], keep_crowd: bool = False):
        """ Loads the given coco files in this CocoSet

        :param files: List of coco files to load (json files)
        :param keep_crowd: If false, discard crowd annotations
        """
        for filename in files:
            file = read_json(filename)
            print("###", filename)
            print("Keys: ", file.keys())
            print("Annotations:", file["annotations"][0].keys())
            print("Images:", file["images"][0].keys(), "\n")

            for image in file.pop("images"):
                image_id = image["id"]
                self.dic[image_id] = self.dic.get(image_id, {}) | image

            for image in file.pop("annotations"):
                image_id = image["image_id"]
                self.dic[image_id]["annotations"] = merge_or_append(self.dic[image_id].get("annotations", []), image)

            for key in file.keys():
                if isinstance(file[key], dict):
                    self.params[key] = self.params.get(key, {}) | file[key]
                else:
                    if isinstance(file[key][0], dict) and key in self.params:
                        for new_v in file[key]:
                            self.params[key] = merge_or_append(self.params.get(key, []), new_v, key='id')
                    else:
                        self.params[key] = self.params.get(key, []) + file[key]

        if not keep_crowd:
            self.filter_crowd()

    def filter_crowd(self):
        """ Removes crowd annotations from the annotations """
        for element in self.dic.values():
            element['annotations'] = [ann for ann in element['annotations'] if ann.get('iscrowd') != 1]

    def load(self, file: str):
        """ Loads the given CocoSet save (json) in this CocoSet """
        file = read_json(file)
        self.params = file["params"]
        self.dic = file["dic"]

    def save(self, output_filename: str):
        """ Saves the current CocoSet in a json file """
        if os.path.isfile(output_filename):
            print("The output file ({}) already exists, "
                  "pls move or remove it before exporting this dataset.".format(output_filename))
            return

        dic = {"params": self.params, "dic": self.dic}
        with open(output_filename, "w") as f:
            json.dump(dic, f, indent=4, cls=NpEncoder)
        print("Saved !")

    def get_new_ann_id(self):
        """ Returns a new id not present in the dataset and adds this id to the dataset. """
        if len(self.ids) == 0:
            self.ids = set(ann.get('id') for element in self.dic.values() for ann in element['annotations'])
            self._available_ids = set(range(max(self.ids))) - self.ids

        if len(self._available_ids) > 0:
            new_id = self._available_ids.pop()
        else:
            new_id = max(self.ids) + 1
        self.ids.add(new_id)
        return new_id

    def get_category(self, category_id: Optional[int] = None, category_name: Optional[str] = None) -> Optional[dict]:
        """ returns the category corresponding to the given id or name """
        categories = self.params['categories']
        for category in categories:
            if category_id is not None and category['id'] == category_id:
                return category
            if category_name is not None and category['name'] == category_name:
                return category
        print("Category not found.")
        return None

    def get_super_category_ids(self, super_category_name: str) -> np.array:
        """ returns all category of the given super_category in a np.array """
        return np.array([cat['id'] for cat in self.params['categories'] if cat['supercategory'] == super_category_name])

    def add_category(self, new_category: dict):
        """
        Assigns an id to his new_category and adds it to this CocoSet

        :param new_category: dict with at list a key: "name"
        """
        if new_category['name'] in [cat['name'] for cat in self.params['categories']]:
            s = "The name of the new category \"{}\" is already in the dataset, add_category() canceled."
            print(s.format(new_category['name']))
            return
        new_id = max([cat['id'] for cat in self.params['categories']]) + 1
        new_category['id'] = new_id
        print("Category \"{}\" added with id {}".format(new_category['name'], new_category['id']))
        self.params['categories'].append(new_category)

    def get_keypoint_dic(self, key_points: List) -> dict:
        """
        Transforms the given List of key_points in a dict associating each keypoint name with its 3 values:
        x, y and visibility. Visibility can take 3 values:
        0 not label (and so x=y=0)
        1 labeled but not visible
        2 labeled and visible
        """
        return {kp: key_points[i * 3: i * 3 + 3] for kp, i in zip(self.params["categories"][0]['keypoints'],
                                                                  range(int(len(key_points) / 3)))}

    def get_images_with_category(self, category_id: int) -> set[int]:
        """ returns the id of all images containing the given category id"""
        return {image_id for image_id, image in self.dic.items()
                for ann in image.get('annotations', [])
                if ann.get('category_id', -1) == category_id}

    def print_image(self, image_id: int,
                    display_bb: bool = True, display_kp: bool = False, display_seg: bool = False,
                    indexes: Optional[List[int]] = None, classes: Optional[List[int]] = None, no_image: bool = False,
                    line_thickness: int = 1):
        """
        Prints the image corresponding to the given id.

        :param image_id: Id of the image to display
        :param display_bb: Display bounding box
        :param display_kp: Display key points
        :param display_seg: Display segmentation
        :param indexes: Display only the annotation at the given indexes
        :param classes: Display only the given list of classes (id)
        :param no_image: Do not display the image (only the annotations)
        :param line_thickness: Thickness of the lines
        """
        if image_id not in self.dic:
            print("No image with the given id () in the current CocoSet.".format(image_id))
            return
        image = cv2.imread(os.path.join(self.image_path, self.dic[image_id]['file_name']))
        if no_image:
            image = np.zeros(image.shape, dtype=image.dtype)

        annotations = self.dic[image_id]['annotations']

        for index in indexes if indexes else range(len(annotations)):
            element = annotations[index]
            if 'category_id' not in element or (classes is not None and element['category_id'] not in classes):
                continue
            if element['category_id'] not in self.colors:
                self.colors[element['category_id']] = [random.randint(0, 255) for _ in range(3)]
            color = self.colors[element['category_id']]
            category = self.get_category(element['category_id'])['name']
            if display_kp and 'keypoints' in element:
                plot_skeleton_kpts(image, element['keypoints'], 3)
            if 'bbox' in element and display_bb:
                bbox = element['bbox']
                bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
                plot_one_box(bbox, image, color=color, label=category, line_thickness=line_thickness)
            if 'segmentation' in element and display_seg:
                plot_segmentation(element['segmentation'], image, color=color)

        # image = letterbox(image, 960, stride=64, auto=True)[0]
        plt.figure(figsize=(8, 8))
        plt.axis('off')
        plt.imshow(image[:, :, [2, 1, 0]])
        plt.show()

    def potentiel_error(self, threshold: float = 3):
        """ Identifies images where the bounding box and segmentation do not appear to be consistent. """
        strange_image = []
        for image_id, image_info in self.dic.items():
            for ann in image_info['annotations']:
                seg = ann.get('segmentation')
                if seg and len(seg) > 1:
                    radius = [segmentation_radius(s) for s in seg]
                    if max(radius) * threshold < distance_between_segmentations(seg):
                        strange_image.append(image_id)
        return strange_image

    def export(self, classes: Optional[List[int]] = None):
        """ Exports the CocoSet data in the format used by YoloV7.

        :param classes: The classes to export (list of ids), default: all
        """
        sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
        annotation_path = self.image_path.replace(sa, sb, 1)
        if os.path.exists(annotation_path):
            print("The annotation directory ({}) already exists, "
                  "pls move or remove it before exporting this dataset.".format(annotation_path))
            return

        os.makedirs(annotation_path, exist_ok=True)
        if classes is None:
            classes = set(self.get_category(ann['category_id'])['name']
                          for image in self.dic.values()
                          for ann in image["annotations"]
                          if 'category_id' in ann)

        mapping_name = {}
        mapping_id = {}
        for category in self.params['categories']:
            if category['name'] in classes:
                mapping_id[category['id']] = len(mapping_id)
                mapping_name[category['name']] = len(mapping_name)

        print("### Classes list:")
        print("nc:", len(mapping_name))
        print("names:", sorted(mapping_name, key=lambda x: mapping_name[x]))

        print("\n### Classes Mapping:")
        print(mapping_name)

        valid_images = []

        print("\nExport")
        for image_id, image in tqdm(self.dic.items()):
            output_name = "{:012d}.txt".format(int(image_id))
            height = image['height']
            width = image['width']
            strings = []
            for ann in image["annotations"]:
                if 'category_id' not in ann or ann['category_id'] not in mapping_id or 'bbox' not in ann:
                    continue
                new_id = mapping_id[ann['category_id']]
                bbox = bbox_coco_to_yolo(ann['bbox'], width, height)
                strings.append("{} {:.6f} {:.6f} {:.6f} {:.6f}".format(new_id, bbox[0], bbox[1], bbox[2], bbox[3]))

            if len(strings):
                with open(os.path.join(annotation_path, output_name), 'w') as f:
                    f.write("\n".join(strings) + "\n")
                valid_images.append(os.path.join(self.image_path, image['file_name']))

        print("Exporting {} images.".format(len(valid_images)))
        with open(os.path.join(annotation_path, "summary.txt"), 'w') as f:
            f.write("\n".join(valid_images) + "\n")

        print("\nList of new dataset images in: {}".format(os.path.join(annotation_path, "summary.txt")))
