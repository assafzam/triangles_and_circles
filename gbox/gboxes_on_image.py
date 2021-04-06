import itertools
from typing import Union, List, Iterator
import matplotlib.pyplot as plt

import numpy as np
import imgaug

import logging
logger = logging.getLogger()

from gbox.gbox import GBox, PredGBox
from gbox.gbox_utils import _remove_out_of_image_fraction_, is_valid_box_dict, is_valid_pred_box_dict, \
    get_ratio_fix
from gbox.types import NumpyImg, ImgHW


class GBoxesOnImage(object):
    """Container for the list of all bounding boxes on a single image.
    Parameters
    ----------
    gboxes : list of green.utils.green_box.GBox
        List of bounding boxes on the image.
    global_data : dict
        additional data of the image (mm_pixel, filepath etc.)
    img_shape : tuple of int (optional)
        The shape of the image on which the objects are placed, i.e. the
        result of ``image.shape``.
        Should include the number of channels, not only height and width.
    Examples
    --------
    >> import numpy
    >> from green.utils.green_box.gbox import GBox, GBoxesOnImage
    >>
    >> image = numpy.zeros((100, 100))
    >> bbs = [
    >>     GBox(x1=10, y1=20, x2=20, y2=30),
    >>     GBox(x1=25, y1=50, x2=30, y2=70)
    >> ]
    >> bbs_oi = GBoxesOnImage(bbs, img_shape=image.shape)
    """

    def __init__(self, gboxes: List[GBox], global_data: dict = None, img_shape: tuple = None):
        self.gboxes:      List[GBox] = gboxes
        self.img_shape:   tuple      = img_shape
        self.global_data: dict       = global_data or {}

    @classmethod
    def from_dicts_array(cls, arr: List[dict], img_shape: tuple=None, global_data: dict=None) -> 'GBoxesOnImage':
        """
        Change the GBoxesOnImage object to all python's builtin objects - for caching and compatibility with old boxes
        Parameters
        ----------
        arr : List of dicts
            each dict supposed to represent a box (with the keys x1,y1,x2,y2, class, etc.)
        img_shape : tuple
            the shape of the image that the boxes belongs to
        global_data : dict
            the additional data of the image (mm_pixel, filepath, etc.)
        Returns
        -------
        GBoxesOnImage
            the GBoxesOnImage object created from the boxes in the dict format

        Examples
        --------
        >> boxes_arr = [{'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'class': 'some_class_1', 'age': 'some_age_1'},
        >>              {'x1': 100, 'y1': 100, 'x2': 100, 'y2': 100, 'class': 'some_class_2'}]
        >> gboxes = GBoxesOnImage.from_dicts_array((boxes_arr))
        """
        if arr:
            return GBoxesOnImage(
                gboxes=[GBox.from_dict(d) for d in arr if is_valid_box_dict(d)],
                img_shape=img_shape,
                global_data=global_data
            )
        else:
            return GBoxesOnImage([])



    @classmethod
    def from_xyxy_array(cls, xyxy: Union[list, np.ndarray], shape: tuple) -> 'GBoxesOnImage':
        """Convert an ``(N, 4)  ndarray`` to a GBoxesOnImage instance.
        This is the inverse of
        :func:`~green.utils.green_box.to_xyxy_array`.
        Parameters
        ----------
        xyxy : (N, 4) ndarray
            Array containing the corner coordinates of ``N`` bounding boxes.
            Each bounding box is represented by its top-left and bottom-right
            coordinates.
            The array should usually be of dtype ``float32``.
        shape : tuple of int
            Shape of the image on which the bounding boxes are placed.
            Should usually be ``(H, W, C)`` or ``(H, W)``.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Object containing a list of :class:`BoundingBox` instances
            derived from the provided corner coordinates.
        """
        xyxy = np.array(xyxy, dtype=np.float32)

        if xyxy.shape[0] == 0:
            return GBoxesOnImage([], img_shape=shape)

        assert xyxy.ndim == 2 and xyxy.shape[-1] == 4, f"Expected input array of shape (N, 4) got shape {xyxy.shape}"

        xyxy = xyxy.reshape((-1, 2, 2))
        boxes = [GBox.from_point_soup(row) for row in xyxy]

        return cls(boxes, img_shape=shape)

    @classmethod
    def from_imgaug_boxes_on_image(
            cls,
            imgaug_boxes: Union[imgaug.BoundingBoxesOnImage,List[imgaug.BoundingBox]],
            img_shape=None
    ) -> 'GBoxesOnImage':

        return GBoxesOnImage(gboxes=[GBox.from_imgaug_box(imgaug_box) for imgaug_box in imgaug_boxes], img_shape=img_shape)

    @property
    def items(self) -> List[GBox]:
        """Get the bounding boxes in this container.
        Returns
        -------
        list of BoundingBox
            Bounding boxes within this container.
        """
        return self.gboxes

    @items.setter
    def items(self, value: List[GBox]):
        """Set the bounding boxes in this container.
        Parameters
        ----------
        value : list of BoundingBox
            Bounding boxes within this container.
        """
        self.gboxes = value

    @property
    def filepath(self) -> Union[str, None]:
        """Get the annotation file absolute path"""
        return self.global_data.get('filepath', None)

    @filepath.setter
    def filepath(self, value: str) -> None:
        """Set the annotation file absolute path"""
        self.global_data['filepath'] = value

    @property
    def mm_pixel(self) -> Union[float, None]:
        """ Get the mm_per_pixel ratio
        The mm_per_pixel ratio helps to get the area of boxes in real world measures.
        Returns
        -------
        number or None
        """
        measure = self.global_data.get('measure', None)
        if measure:
            return measure.get('mm_pixel', None)
        else:
            return None

    @mm_pixel.setter
    def mm_pixel(self, value: float) -> None:
        """Set the mm_per_pixel value
        The mm_per_pixel ratio helps to get the area of boxes in real world measures.

        """
        measure = self.global_data.get('measure', None)
        if measure:
            measure['mm_pixel'] = value
        else:
            self.global_data['measure'] = {'mm_pixel': value}

    @property
    def empty(self) -> bool:
        """Determine whether this instance contains zero bounding boxes.
        Returns
        -------
        bool
            True if this object contains zero bounding boxes.
        """
        return len(self.gboxes) == 0

    def get_classes_names(self) -> set:
        """Finds all classes of the gboxes
        Returns
        -------
        set of str. each str is a class name
        """
        classes_names = {box.class_name for box in self.gboxes}
        try:
            classes_names.remove(None)
        except KeyError:
            pass
        return classes_names



    def to_dicts_array(self, cast_to_int=False) -> List[dict]:
        """
        Convert the GBoxesOnImage to list of dict, each dict is box (with keys x1,x2.y1,y2, class, etc.)
        This is the inverse of
        :func:`~green.utils.green_box.from_dicts_array`.
        Returns
        -------
        list of dicts
            each dict is box (with keys x1,x2.y1,y2, class, etc.)
        """
        return [gbox.to_dict(cast_to_int) for gbox in self.gboxes]

    def to_xyxy_array(self, dtype=np.float32) -> np.ndarray:
        """Convert the ``BoundingBoxesOnImage`` object to an ``(N,4) ndarray``.
        This is the inverse of
        :func:`~green.utils.green_box.from_xyxy_array`.
        Parameters
        ----------
        dtype : numpy.dtype, optional
            Desired output datatype of the ndarray.
        Returns
        -------
        ndarray
            ``(N,4) ndarray``, where ``N`` denotes the number of bounding
            boxes and ``4`` denotes the top-left and bottom-right bounding
            box corner coordinates in form ``(x1, y1, x2, y2)``.
        """
        xyxy_array = np.zeros((len(self.gboxes), 4), dtype=np.float32)

        for i, box in enumerate(self.gboxes):
            xyxy_array[i] = [box.x1, box.y1, box.x2, box.y2]

        return xyxy_array.astype(dtype)


    def to_imgaug_boxes_on_image(self, shape) -> imgaug.BoundingBoxesOnImage:
        """
        convert green boxes to imgaug boxes
        :param shape: the image (who the boxes belong to) shape.
        :type shape: numpy array (w,h,c)
        :return: imgaug.BoundingBoxesOnImage object holding the boxes (each box is imgaug.BoundingBox object)
        """
        imgaug_boxes = [
            gbox.to_imgaug_box()
            for gbox in self.gboxes
        ]
        return imgaug.BoundingBoxesOnImage(imgaug_boxes, shape)

    def avg_iou(self, other_boxes, k):
        accuracy = np.mean([np.max(self.iou(other_boxes, k), axis=1)])
        return accuracy

    def iou(self, other_boxes: 'GBoxesOnImage', k: int) -> np.ndarray:
        """
        Calculating the iou of 2 GBoxesOnImage objects.

        """
        boxes = self.to_xyxy_array()
        other_boxes = other_boxes.to_xyxy_array()

        n = len(boxes)

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = other_boxes[:, 0] * other_boxes[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(other_boxes[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(other_boxes[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)

        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def on_(self, from_shape, to_shape) -> 'GBoxesOnImage':
        """Project BBs from one image (shape) to a another one in-place.
        Parameters
        ----------
        from_shape : tuple of int
            The original shape of image which the bounding boxes are projected from.
        to_shape : tuple of int
            The target shape of image onto which the bounding boxes are to be projected.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Object containing the same bounding boxes after projection to
            the new image shape.
            The object and its items may have been modified in-place.
        """
        if to_shape[0:2] == from_shape[0:2]:
            self.img_shape = to_shape  # channels may differ
            return self

        for i, item in enumerate(self.items):
            self.gboxes[i] = item.project_(from_shape, to_shape)
        self.img_shape = to_shape
        return self

    def on(self, from_shape, to_shape) -> 'GBoxesOnImage':
        """Project bounding boxes from one image (shape) to a another one.
        Parameters
        ----------
        from_shape : tuple of int
            The original shape of image which the bounding boxes are projected from.
        to_shape : tuple of int
            The target shape of image onto which the bounding boxes are to be projected.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Object containing the same bounding boxes after projection to
            the new image shape.
        """
        # pylint: disable=invalid-name
        return self.deepcopy().on_(from_shape, to_shape)


    def draw_on_image(self, image: NumpyImg, single_color=None, class_to_color=None, thickness=2, show=False, in_place=False):

        """Draw all bounding boxes onto a given image.
        Parameters
        ----------
        image : (H,W,3) ndarray
            The image onto which to draw the bounding boxes.
            This image should usually have the same shape as set in
            ``BoundingBoxesOnImage.shape``.
        class_to_color : dict or None
            map from class name to color.
            If a single ``int`` ``C``, then that is equivalent to ``(C,C,C)``.
D
        thickness : None or int, optional
            The size of the line of the box.
        show : bool, optional, default is False
            if True plotting the image with matplotlib plt.show()
        Returns
        -------
        (H,W,3) ndarray
            Image with drawn bounding boxes.
        """
        import logging
        logging.getLogger('cv2').setLevel(logging.WARNING)

        if self.empty:
            return image
        if not in_place:
            image = np.copy(image)

        if class_to_color is None:
            def get_spaced_colors(bboxes):
                n = len(bboxes)
                max_value = 16581375  # 255**3
                interval = int(max_value / n)
                colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]

                rgb_colors = [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]
                return dict(zip([b.class_name for b in bboxes], np.array(rgb_colors)))

            color_map = get_spaced_colors(self.gboxes)
            class_to_color = lambda x: color_map[x]
        if single_color is not None:
            class_to_color = lambda x: single_color

        _ = [gbox.draw_on_image(image=image, color=class_to_color(gbox.class_name), thickness=thickness, in_place=True)
             for gbox in self.gboxes]
        if show:
            plt.imshow(image)
            plt.show()

        return image

    def remove_out_of_image_(self, fully=True, partly=False) -> 'GBoxesOnImage':
        """Remove in-place all BBs that are fully/partially outside of the image.
        Added in 0.4.0.
        Parameters
        ----------
        fully : bool, optional
            Whether to remove bounding boxes that are fully outside of the
            image.
        partly : bool, optional
            Whether to remove bounding boxes that are partially outside of
            the image.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Reduced set of bounding boxes, with those that were
            fully/partially outside of the image being removed.
            The object and its items may have been modified in-place.
        """
        if self.img_shape:
            self.gboxes = [
                bb
                for bb
                in self.gboxes
                if not bb.is_out_of_image(self.img_shape, fully=fully, partly=partly)]
            return self
        else:
            logger.warning("cannot remove BBs out of image Because there is no shape!, returns original BBs")
            return self

    def remove_out_of_image(self, fully=True, partly=False) -> 'GBoxesOnImage':
        """Remove all BBs that are fully/partially outside of the image.
        Parameters
        ----------
        fully : bool, optional
            Whether to remove bounding boxes that are fully outside of the
            image.
        partly : bool, optional
            Whether to remove bounding boxes that are partially outside of
            the image.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Reduced set of bounding boxes, with those that were
            fully/partially outside of the image being removed.
        """
        return self.copy().remove_out_of_image_(fully=fully, partly=partly)

    def remove_out_of_image_fraction_(self, fraction) -> 'GBoxesOnImage':
        """Remove in-place all BBs with an OOI fraction of at least `fraction`.
        'OOI' is the abbreviation for 'out of image'.
        Added in 0.4.0.
        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a bounding box has to have in
            order to be removed. A fraction of ``1.0`` removes only bounding
            boxes that are ``100%`` outside of the image. A fraction of ``0.0``
            removes all bounding boxes.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Reduced set of bounding boxes, with those that had an out of image
            fraction greater or equal the given one removed.
            The object and its items may have been modified in-place.
        """
        return _remove_out_of_image_fraction_(self, fraction)

    def remove_out_of_image_fraction(self, fraction) -> 'GBoxesOnImage':
        """Remove all BBs with an out of image fraction of at least `fraction`.
        Parameters
        ----------
        fraction : number
            Minimum out of image fraction that a bounding box has to have in
            order to be removed. A fraction of ``1.0`` removes only bounding
            boxes that are ``100%`` outside of the image. A fraction of ``0.0``
            removes all bounding boxes.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Reduced set of bounding boxes, with those that had an out of image
            fraction greater or equal the given one removed.
        """
        return self.copy().remove_out_of_image_fraction_(fraction)

    def clip_out_of_image_(self, img_shape: ImgHW=None) -> 'GBoxesOnImage':
        """
        Clip off in-place all parts from all BBs that are outside of the image.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Bounding boxes, clipped to fall within the image dimensions.
            The object and its items may have been modified in-place.
        """
        shape = img_shape if img_shape is not None else self.img_shape

        if shape:
            # remove bbs that are not at least partially inside the image plane
            self.gboxes = [bb for bb in self.gboxes
                           if bb.is_partly_within_image(shape)]

            for i, bb in enumerate(self.gboxes):
                self.gboxes[i] = bb.clip_out_of_image(shape)

            return self
        else:
            logger.warning("cannot clip out BBs of image Because there is no shape!, returns original BBs")
            return self

    def clip_out_of_image(self) -> 'GBoxesOnImage':
        """Clip off all parts from all BBs that are outside of the image.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Bounding boxes, clipped to fall within the image dimensions.
        """
        return self.deepcopy().clip_out_of_image_()

    def shift_(self, x=0, y=0) -> 'GBoxesOnImage':
        """Move all BBs along the x/y-axis in-place.
        The origin ``(0, 0)`` is at the top left of the image.
        Added in 0.4.0.
        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.
        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Shifted bounding boxes.
            The object and its items may have been modified in-place.
        """
        for i, bb in enumerate(self.gboxes):
            self.gboxes[i] = bb.shift_(x=x, y=y)
        return self

    def shift(self, x=0, y=0) -> 'GBoxesOnImage':
        """Move all BBs along the x/y-axis.
        The origin ``(0, 0)`` is at the top left of the image.
        Parameters
        ----------
        x : number, optional
            Value to be added to all x-coordinates. Positive values shift
            towards the right images.
        y : number, optional
            Value to be added to all y-coordinates. Positive values shift
            towards the bottom images.

        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Shifted bounding boxes.
        """
        return self.deepcopy().shift_(x=x, y=y)

    def copy(self, bounding_boxes=None, shape=None) -> 'GBoxesOnImage':
        """Create a shallow copy of the ``BoundingBoxesOnImage`` instance.
        Parameters
        ----------
        bounding_boxes : None or list of green.utils.green_box.GBox, optional
            List of bounding boxes on the image.
            If ``None``, the instance's bounding boxes will be copied.
        shape : tuple of int, optional
            The shape of the image on which the bounding boxes are placed.
            If ``None``, the instance's shape will be copied.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Shallow copy.
        """
        if bounding_boxes is None:
            bounding_boxes = self.gboxes[:]
        if shape is None:
            shape = self.img_shape
        else:
            # use tuple() here in case the shape was provided as a list
            shape=tuple(shape)

        return GBoxesOnImage(bounding_boxes, img_shape=shape)

    def deepcopy(self, bounding_boxes=None, shape=None, global_data=None) -> 'GBoxesOnImage':
        """Create a deep copy of the ``BoundingBoxesOnImage`` object.
        Parameters
        ----------
        bounding_boxes : None or list of green.utils.green_box.GBox, optional
            List of bounding boxes on the image.
            If ``None``, the instance's bounding boxes will be copied.
        shape : tuple of int, optional
            The shape of the image on which the bounding boxes are placed.
            If ``None``, the instance's shape will be copied.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Deep copy.
        """
        # Manual copy is far faster than deepcopy, so use manual copy here.
        if bounding_boxes is None:
            bounding_boxes = [bb.deepcopy() for bb in self.gboxes]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = self.img_shape
        if global_data is None:
            global_data = self.global_data

        return GBoxesOnImage(bounding_boxes, img_shape=shape, global_data=global_data)

    def fix_boxes_by_order_operation(self, order_operation: List[dict], remove_out_of_bound: bool=True)->'GBoxesOnImage':
        return self.deepcopy().fix_boxes_by_order_operation_(order_operation, remove_out_of_bound=True)

    def fix_boxes_by_order_operation_(self, order_operation: List[dict], remove_out_of_bound: bool=True)->'GBoxesOnImage':

        for o in order_operation:
            command = o.get("command", None)

            if command == "resize":
                resized_height, resized_width = o.get("value")
                original_height, original_width = o.get("base")
                self.on_(from_shape=(original_width,original_height),
                          to_shape=(resized_width, resized_height))

            elif command == "crop":
                top, bottom, left, right = o.get('value')

                self.fix_boxes_cropped_(top, bottom, left, right, remove_out_of_bound=remove_out_of_bound)
            elif command == "padding_wrap":
                self.fix_boxes_padding_wrap_(orig_img_size=o.get('base'), pad_ratio=o.get('value'))
            elif command == "padding":
                top, left = o.get('offset')
                if top != 0 or left != 0:
                    self.fix_boxes_cropped_(top, 0, left, 0)
            elif command == "rotate":
                degree = o.get('value')
                height, width = o.get('base')
                assert degree == 90
                self.fix_boxes_rotate_(width)

        self.clean_zero_length_boxes_()
        return self

    def fix_boxes_cropped_(self, top: float, bottom: float, left: float, right: float,
                          threshold: float = 0.5, remove_out_of_bound: bool = True):
        """
        remove boxes out of image.
        otherwise if part of the box is in the area then just align it to the new top,left
        note that it means the boxes can be still partly out of the image
        :param threshold:
        :param boxes: GBoxesOnImage object
        :param top:
        :param bottom:
        :param left:
        :param right:
        :return: GBoxesOnImage object
        """
        to_remove = []
        epsilon = 1
        for i, b in enumerate(self.gboxes):
            b_top = b.y1
            b_bottom = b.y2
            b_left = b.x1
            b_right = b.x2
            if remove_out_of_bound:
                if (b_right < left) or (b_bottom < top) or (b_left > right) or (b_top > bottom):
                    to_remove.append(i)
                    continue
                # remove flat boxes
                if (b_bottom - b_top) < epsilon or (b_right - b_left) < epsilon:
                    to_remove.append(i)
                    continue
                # remove box if its enire
                h = np.min([b_bottom, bottom]) - np.max([top, b_top])
                w = np.min([b_right, right]) - np.max([left, b_left])
                part_in_crop = w * h / ((b_bottom - b_top) * (b_right - b_left) * 1.0)
                if part_in_crop < threshold:
                    to_remove.append(i)
                    continue

            b.x1 -= left
            b.y1 -= top
            b.x2 -= left
            b.y2 -= top

        for index in sorted(to_remove, reverse=True):
            del self.gboxes[index]
        return self

    def fix_boxes_padding_wrap_(self, orig_img_size: tuple, pad_ratio: tuple) -> 'GBoxesOnImage':
        """
        the method get as input boxes of image after the image was duplicated to fit the target size and target mm_p_pixel,
        and fix the boxes:
            duplicates the bboxes and shift them to the new position.

        for example:

        <--orig_size-->                           <-------------target size ------------>
                                        index:     [0,0]        [0,1]        [0,2]
        |------------|                           |------------|------------|------------|
        |         [] |      pad_ratio=(1,3)      |         [] |         [] |         [] |
        |  orig img  |    ------------------>    |  orig img  |  copy 1    |  copy 2    |
        |  []        |                           |   []       |   []       |   []       |
        |------------|                           |------------|------------|------------|

        ** [] is bbox

        :param orig_img_size: tuple of integers (h,w) of the orig img size
        :param pad_ratio: tuple of integers (h,w) that says how many times the image were duplicated to each direction
        :return:
        """
        height, width = orig_img_size
        pad_height_ratio, pad_width_ratio = pad_ratio

        #  iterating on the dot product of pad_height_ratio and pad_width_ratio
        #    for example: pad_ratio = (1,3) --> iterating on  = [(0,0), (0,1), (0,2)]
        new_boxes = [
            box.deepcopy().shift_(x=j * width, y=i * height)
            for box in self.gboxes
            for i, j in itertools.product(range(pad_height_ratio), range(pad_width_ratio))
        ]
        self.gboxes = new_boxes
        return self

    def fix_boxes_rotate_(self, img_width) -> 'GBoxesOnImage':
        for i, b in enumerate(self.gboxes):
            before_box = b.copy()
            b.x1 = before_box.y1
            b.y1 = img_width - before_box.x2
            b.x2 = before_box.y2
            b.y2 = img_width - before_box.x1
        return self

    def clean_zero_length_boxes_(self) -> None:
        """ removes (in place!) all the empty boxes"""
        self.gboxes = [box for box in self.items if not box.empty]

    @staticmethod
    def squeeze(boxes: Union[List['GBoxesOnImage'], 'GBoxesOnImage', None]):
        """ if boxes is a single box in a list container (python list or np.array, extracting the box out side
        from the container and returning a single GBox object, else returning the boxes as received."""
        if boxes:
            if isinstance(boxes, (list, np.ndarray)):
                if len(boxes) == 1 and isinstance(boxes[0], GBoxesOnImage):
                    return boxes[0]
        return boxes


    def __getitem__(self, indices: Union[float, slice]) -> Union[GBox, List[GBox]]:
        """Get the bounding box(es) with given indices.
        Returns
        -------
        list of green.utils.green_box.GBox
            Bounding box(es) with given indices.
        """
        return self.gboxes[indices]

    def __iter__(self) -> Iterator[GBox]:
        """Iterate over the bounding boxes in this container.
        Yields
        ------
        BoundingBox
            A bounding box in this container.
            The order is identical to the order in the bounding box list
            provided upon class initialization.
        """
        return iter(self.gboxes)

    def __len__(self):
        """Get the number of items in this instance.
        Returns
        -------
        int
            Number of items in this instance.
        """
        return len(self.items)

    def __repr__(self):
        return self.__str__()

    def __delitem__(self, key):
        del self.gboxes[key]

    def __str__(self):
        return f"GBoxesOnImage({str(self.gboxes)}, img_shape={self.img_shape})"

    def __bool__(self):
        return len(self.gboxes) > 0

    def __hash__(self):
        return hash(tuple(self.gboxes)) + hash((self.img_shape, self.mm_pixel))

    def __eq__(self, other):
        """
        Determine if 2 GBoxesOnImage are Equal:
        They equal iff they have the same boxes - as sets.
        and that they have the same shape
        :param other:
        :return:
        """
        if isinstance(other, self.__class__):
            if (set(self.gboxes) == set(other.gboxes)) and (self.img_shape == other.img_shape):
                return True
        return False


class PredGBoxesOnImage(GBoxesOnImage):
    def __init__(self, gboxes: List[PredGBox], global_data: dict = None, img_shape: tuple = None):
        super().__init__(gboxes, global_data, img_shape)
        self.gboxes: List[PredGBox] = gboxes

    @classmethod
    def from_dicts_array(cls, arr: List[dict], img_shape: tuple=None, global_data: dict=None) -> 'PredGBoxesOnImage':
        """
        Change the PredGBoxesOnImage object to all python's builtin objects - for caching and compatibility with old boxes
        Parameters
        ----------
        arr : List of dicts
            each dict supposed to represent a box (with the keys x1,y1,x2,y2, class, etc.)
        img_shape : tuple
            the shape of the image that the boxes belongs to
        global_data : dict
            the additional data of the image (mm_pixel, filepath, etc.)
        Returns
        -------
        GBoxesOnImage
            the GBoxesOnImage object created from the boxes in the dict format

        Examples
        --------
        >> boxes_arr = [{'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'probs': {'class_name': [1,2,3,4]},
        >>               'class': 'some_class_1', 'age': 'some_age_1'},
        >>              {'x1': 100, 'y1': 100, 'x2': 100, 'y2': 100,'probs': 3.4},
        >>               'class': 'some_class_2'}]
        >> gboxes = PredGBoxesOnImage.from_dicts_array((boxes_arr))
        """
        if arr:
            return PredGBoxesOnImage(
                gboxes=[PredGBox.from_dict(d) for d in arr if is_valid_pred_box_dict(d)],
                img_shape=img_shape,
                global_data=global_data
            )
        else:
            return PredGBoxesOnImage([])

    def fix_boxes_post_prediction(self, order_operation: List[dict]) -> 'PredGBoxesOnImage':
        return self.deepcopy().fix_boxes_post_prediction_(order_operation)

    def fix_boxes_post_prediction_(self, order_operation: List[dict]) -> 'PredGBoxesOnImage':
        """ Fix The boxes By the order operation - after prediction.
        """

        order_operation = reversed(order_operation)

        for o in order_operation:
            if "command" in o:
                command = o.get("command")

                if command == "resize":
                    resized_height, resized_width = o.get("value")
                    original_height, original_width = o.get("base")
                    self._fix_boxes_resize_post_prediction_(width=original_width, height=original_height,
                                                            resized_width=resized_width, resized_height=resized_height)
                elif command == "crop":
                    top, bottom, left, right = o.get('value')
                    self._fix_boxes_cropped_post_prediction_(top, bottom, left, right)
                elif command == "padding":
                    top, left = o.get('offset')
                    if top != 0 or left != 0:
                        self._fix_boxes_cropped_post_prediction_(top, 0, left, 0)
                elif command == "rotate":
                    degree = o.get('value')
                    height, width = o.get('base')
                    assert degree == 90
                    self._fix_boxes_rotate_post_prediction_(width)
        return self

    def _fix_boxes_rotate_post_prediction_(self, width):
        for box in self.gboxes:
            x1 = np.copy(box.x1)
            y1 = np.copy(box.y1)
            x2 = np.copy(box.x2)
            y2 = np.copy(box.y2)

            box.x1 = width - y2
            box.y1 = x1
            box.x2 = width - y1
            box.y2 = x2

    def _fix_boxes_resize_post_prediction_(self, width, height, resized_width, resized_height):
        """
        Fix all coordinates. this can be done without loopping on numpy array
        :param bboxes:dict. the dict format is {'effective_class_str': [[x1,y1,x2,y2], ...](np.array)}.
        :return:
        """

        h_fix, w_fix = get_ratio_fix(o_width=width, o_height=height, resized_width=resized_width,
                                     resized_height=resized_height)
        for box in self.gboxes:
            box.x1 = int(box.x1*w_fix)
            box.y1 = int(box.y1*h_fix)
            box.x2 = int(box.x2*w_fix)
            box.y2 = int(box.y2*h_fix)

    def _fix_boxes_cropped_post_prediction_(self, top, bottom, left, right):
        """
        remove boxes out of image.
        otherwise if part of the box is in the area then just align it to the new top,left
        note that it means the boxes can be still partly out of the image
        """
        for box in self.gboxes:
            box.x1 += left
            box.y1 += top
            box.x2 += left
            box.y2 += top

    def __iter__(self) -> Iterator[PredGBox]:
        """Iterate over the bounding boxes in this container.
        Yields
        ------
        BoundingBox
            A bounding box in this container.
            The order is identical to the order in the bounding box list
            provided upon class initialization.
        """
        return iter(self.gboxes)

    def copy(self, bounding_boxes=None, shape=None) -> 'PredGBoxesOnImage':
        """Create a shallow copy of the ``BoundingBoxesOnImage`` instance.
        Parameters
        ----------
        bounding_boxes : None or list of green.utils.green_box.GBox, optional
            List of bounding boxes on the image.
            If ``None``, the instance's bounding boxes will be copied.
        shape : tuple of int, optional
            The shape of the image on which the bounding boxes are placed.
            If ``None``, the instance's shape will be copied.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Shallow copy.
        """
        if bounding_boxes is None:
            bounding_boxes = self.gboxes[:]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = tuple(self.img_shape)

        return PredGBoxesOnImage(bounding_boxes, img_shape=shape)

    def deepcopy(self, bounding_boxes=None, shape=None, global_data=None) -> 'PredGBoxesOnImage':
        """Create a deep copy of the ``BoundingBoxesOnImage`` object.
        Parameters
        ----------
        bounding_boxes : None or list of green.utils.green_box.GBox, optional
            List of bounding boxes on the image.
            If ``None``, the instance's bounding boxes will be copied.
        shape : tuple of int, optional
            The shape of the image on which the bounding boxes are placed.
            If ``None``, the instance's shape will be copied.
        Returns
        -------
        green.utils.green_box.GBoxesOnImage
            Deep copy.
        """
        # Manual copy is far faster than deepcopy, so use manual copy here.
        if bounding_boxes is None:
            bounding_boxes = [bb.deepcopy() for bb in self.gboxes]
        if shape is None:
            # use tuple() here in case the shape was provided as a list
            shape = self.img_shape
        if global_data is None:
            global_data = self.global_data

        return PredGBoxesOnImage(bounding_boxes, img_shape=shape, global_data=global_data)

    def filter_below_prob(self, threshold_prob: float) -> 'PredGBoxesOnImage':
        return self.deepcopy().filter_below_prob_(threshold_prob)

    def filter_below_prob_(self, threshold_prob: float) -> 'PredGBoxesOnImage':
        """Filter out, IN PLACE,  all the boxes with top prob bellow @param threshold_prob
        Parameters
        ----------
        threshold_prob (float)
            the threshold probability
        Returns
        -------
        PredGBoxesOnImage
            self, without the boxes with top_prob<=threshold_prob

        """
        if not isinstance(threshold_prob, (float,int)):
            logger.warning(f"PredGBoxesOnImage.filter_below_prob_ got threshold_prob parameter from"
                           f" type {type(threshold_prob)}, and expecting to be number. Continuing without filtering")
        else:
            self.gboxes = [gbox for gbox in self.gboxes if gbox.top_prob >= threshold_prob]
        return self


    def non_max_suppression_(self, overlapThresh: float=0.2) -> 'PredGBoxesOnImage':
        # if there are no boxes, return an empty list
        if self.empty:
            return PredGBoxesOnImage([])

        idxs = np.argsort([gbox.top_prob for gbox in self.gboxes])
        x1 = np.array([gbox.x1 for gbox in self.gboxes])
        y1 = np.array([gbox.y1 for gbox in self.gboxes])
        x2 = np.array([gbox.x2 for gbox in self.gboxes])
        y2 = np.array([gbox.y2 for gbox in self.gboxes])

        # compute the area of the bounding boxes and grab the indexes to sort
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # initialize the list of picked indexes
        pick = []

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the index value
            # to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of the bounding
            # box and the smallest (x, y) coordinates for the end of the bounding
            # box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have overlap greater
            # than the provided overlap threshold
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked
        new_gboxes = []
        for i in pick:
            new_gboxes.append(self.gboxes[i])
        self.gboxes = new_gboxes
        return self





if __name__ == '__main__':
    boxes = PredGBoxesOnImage([PredGBox(), PredGBox()])
    for box in boxes:
        print(type(box))
        cop = box.copy()
        dcop = box.deepcopy()