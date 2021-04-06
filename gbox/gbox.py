import copy
from collections import OrderedDict
from typing import Union, Tuple, List, Any

import numpy as np
import imgaug
import cv2

import matplotlib.pyplot as plt
import logging
logger = logging.getLogger()

from gbox.gbox_utils import project_coords, normalize_imglike_shape, is_np_array, \
    is_iterable, flatten, convert_pixels_2_meter, verify_gbox_params


class GBox(object):
    """Class representing our green boxes.
    Each bounding box is parameterized by its top left and bottom right
    corners. Both are given as x and y-coordinates. The corners are intended
    to lie inside the bounding box area. As a result, a bounding box that lies
    completely inside the image but has maximum extensions would have
    coordinates ``(0.0, 0.0)`` and ``(W - epsilon, H - epsilon)``. Note that
    coordinates are saved internally as floats.
    Parameters
    ----------
    x1 : number
        X-coordinate of the top left of the bounding box.
    y1 : number
        Y-coordinate of the top left of the bounding box.
    x2 : number
        X-coordinate of the bottom right of the bounding box.
    y2 : number
        Y-coordinate of the bottom right of the bounding box.
    metadata : None or dict, optional
        The metadata of the bounding box
    class_name : str or None, optional
        The class of the box


    Note
    ----
    1. if class_name is None, and the @param meta_data have a key 'class', the class_name of the box will the value,
       otherwise class_name is @param class_name
    2. for having effective classes names set up costume data structure in the green.config.context object.

    """

    def __init__(
            self,
            x1                   : float = 0,
            y1                   : float = 0,
            x2                   : float = 0,
            y2                   : float = 0,
            metadata             : dict  = None,
            class_name           : str   = None,
    ):
        """Create a new GBox instance."""
        verify_gbox_params(locals())



        if x1 > x2:
            x2, x1 = x1, x2
        if y1 > y2:
            y2, y1 = y1, y2

        self.x1: float = x1
        self.y1: float = y1
        self.x2: float = x2
        self.y2: float = y2
        if isinstance(metadata, dict):
            self.class_name: str = class_name or metadata.get('class')
            self.metadata: dict = {k: v for k,v in metadata.items() if k != 'class'}
        else:
            self.class_name = class_name
            self.metadata: dict = {}



    @property
    def empty(self) -> bool:
        """decide if a box is empty or not.
        GBox is empty iff at least one of the edges of the box has zero length.
        i.e box is empty <--> (x1 - x2 == 0) or (y1 - y2 == 0).
        Returns
        -------
        bool
            True if empty otherwise False

        """
        return (self.x2 - self.x1 == 0) or (self.y2 - self.y1 == 0)



    def almost_empty(self, epsilon: float):
        """decide if a box is almost empty or not.
        GBox is almost empty (with respect to epsilon) iff on of the edges of the box has length smaller than epsilon.
        i.e box is almost empty (with respect to epsilon) <--> (x1 - x2 < epsilon) or (y1 - y2 < epsilon).
        Parameters
        ----------
        epsilon : number
            The threshold length of edge, that if a box have edge shorter then epsilon, the box is almost empty.

        Returns
        -------
        bool
            True if is almost empty (with respect to epsilon) otherwise False

        """
        return (abs(self.x2 - self.x1) < epsilon) or (abs(self.y1 - self.y2) < epsilon)

    @property
    def coords(self) -> np.ndarray:
        """Get the top-left and bottom-right coordinates as one array.
        Returns
        -------
        ndarray
            A ``(N, 2)`` numpy array with ``N=2`` containing the top-left
            and bottom-right coordinates.
        """
        arr = np.empty((2, 2), dtype=np.float32)
        arr[0, :] = (self.x1, self.y1)
        arr[1, :] = (self.x2, self.y2)
        return arr

    @property
    def xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2])

    @property
    def label(self) -> str:
        """The label to be shown in printings, plotting etc.

        Returns
        -------
        str
            the class name of the box if it's not None else empty string ""
        """

        return self.class_name or ""

    @property
    def full_label(self) -> str:
        """The full label of the box constructed from the effective class name and the class name.
        for example if the effective class name is `Broad Leaf 1-2` , and the class name is 'Scientific name`
        the result will be `Broad Leaf 1-2(Scientific name)`
        if there isn't class name the result will be the self.class_name.
        """
        return self.label

    @property
    def x1_int(self) -> int:
        """Get the x-coordinate of the top left corner as an integer.
        Returns
        -------
        int
            X-coordinate of the top left corner, rounded to the closest
            integer.
        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.x1))

    @property
    def y1_int(self) -> int:
        """Get the y-coordinate of the top left corner as an integer.
        Returns
        -------
        int
            Y-coordinate of the top left corner, rounded to the closest
            integer.
        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.y1))

    @property
    def x2_int(self) -> int:
        """Get the x-coordinate of the bottom left corner as an integer.
        Returns
        -------
        int
            X-coordinate of the bottom left corner, rounded to the closest
            integer.
        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.x2))

    @property
    def y2_int(self) -> int:
        """Get the y-coordinate of the bottom left corner as an integer.
        Returns
        -------
        int
            Y-coordinate of the bottom left corner, rounded to the closest
            integer.
        """
        # use numpy's round to have consistent behaviour between python
        # versions
        return int(np.round(self.y2))

    @property
    def height(self) -> float:
        """Estimate the height of the bounding box.
        Returns
        -------
        number
            Height of the bounding box.
        """
        return self.y2 - self.y1

    @property
    def width(self) -> float:
        """Estimate the width of the bounding box.
        Returns
        -------
        number
            Width of the bounding box.
        """
        return self.x2 - self.x1

    @property
    def center_x(self) -> float:
        """Estimate the x-coordinate of the center point of the bounding box.
        Returns
        -------
        number
            X-coordinate of the center point of the bounding box.
        """
        return self.x1 + self.width / 2

    @property
    def center_y(self) -> float:
        """Estimate the y-coordinate of the center point of the bounding box.
        Returns
        -------
        number
            Y-coordinate of the center point of the bounding box.
        """
        return self.y1 + self.height / 2

    @property
    def area(self) -> float:
        """Estimate the area of the bounding box.
        Returns
        -------
        number
            Area of the bounding box, i.e. ``height * width``.
        """
        return self.height * self.width

    @property
    def size(self) -> tuple:
        """Get the size of the box in the form (H,W)
        Returns
        -------
        tuple(float,float)
            Size of the bounding box, i.e. ``(width,height)``.
        """
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """
        calculate the aspect ratio of the box: min(width, height) / max(width, height)
        Returns
        -------
         (float) in range [0,1] or -1 in case of error
        """
        try:
            ratio = min(self.width, self.height) / max(self.width, self.height)
        except ZeroDivisionError:
            ratio = -1
        return ratio

    def square_meter(self, mm_per_pixel: float) -> float:
        """
        calculate the box size in square meter with respect to the mm_per_pixel.
        Parameters
        ----------
        mm_per_pixel: float represents the mm per pixel ratio

        Retrurns
        --------
        float
            the area of the box in square meter with respect to mm per pixel ratio.
        """
        return convert_pixels_2_meter(self.width, mm_per_pixel) * convert_pixels_2_meter(self.height, mm_per_pixel)


    def contains(self, point: tuple) -> bool:
        """Estimate whether the bounding box contains a given point.
        Parameters
        ----------
        point : tuple of number Point to check for.
        Returns
        -------
        bool
            ``True`` if the point is contained in the bounding box,
            ``False`` otherwise.
        """
        x, y = point
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def project_(self, from_shape: Union[Tuple[int], np.ndarray], to_shape: Union[Tuple[int], np.ndarray]):
        """Project the bounding box onto a differently shaped image in-place.
        E.g. if the bounding box is on its original image at
        ``x1=(10 of 100 pixels)`` and ``y1=(20 of 100 pixels)`` and is
        projected onto a new image with size ``(width=200, height=200)``,
        its new position will be ``(x1=20, y1=40)``.
        (Analogous for ``x2``/``y2``.)
        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).
        Parameters
        ----------
        from_shape : tuple of int or ndarray
            Shape of the original image. (Before resize.)
        to_shape : tuple of int or ndarray
            Shape of the new image. (After resize.)
        Returns
        -------
        green.utils.green_box.GBox
            ``GBox`` instance with new coordinates.
            The object may have been modified in-place.
        """
        (self.x1, self.y1), (self.x2, self.y2) = project_coords(
            coords=[(self.x1, self.y1), (self.x2, self.y2)],
            from_shape=from_shape,
            to_shape=to_shape)
        return self

    def project(self, from_shape: Union[np.ndarray, tuple], to_shape: Union[np.ndarray, tuple]):
        """Project the bounding box onto a differently shaped image.
        E.g. if the bounding box is on its original image at
        ``x1=(10 of 100 pixels)`` and ``y1=(20 of 100 pixels)`` and is
        projected onto a new image with size ``(width=200, height=200)``,
        its new position will be ``(x1=20, y1=40)``.
        (Analogous for ``x2``/``y2``.)
        This is intended for cases where the original image is resized.
        It cannot be used for more complex changes (e.g. padding, cropping).
        Parameters
        ----------
        from_shape : tuple of int or ndarray
            Shape of the original image. (Before resize.)
        to_shape : tuple of int or ndarray
            Shape of the new image. (After resize.)
        Returns
        -------
        green.utils.green_box.GBox instance with new coordinates.
        """
        return self.deepcopy().project_(from_shape, to_shape)

    def extend_(self, all_sides: float = 0, top: float = 0, right: float = 0, bottom: float = 0, left: float = 0):
        """Extend the size of the bounding box along its sides in-place.
        Parameters
        ----------
        all_sides : number, optional
            Value by which to extend the bounding box size along all
            sides.
        top : number, optional
            Value by which to extend the bounding box size along its top
            side.
        right : number, optional
            Value by which to extend the bounding box size along its right
            side.
        bottom : number, optional
            Value by which to extend the bounding box size along its bottom
            side.
        left : number, optional
            Value by which to extend the bounding box size along its left
            side.
        Returns
        -------
        green.utils.green_box.GBox
        Extended bounding box.
        The object may have been modified in-place.
        """
        self.x1 = self.x1 - all_sides - left
        self.x2 = self.x2 + all_sides + right
        self.y1 = self.y1 - all_sides - top
        self.y2 = self.y2 + all_sides + bottom
        return self

    def extend(self, all_sides: float = 0, top: float = 0, right: float = 0, bottom: float = 0, left: float = 0):
        """Extend the size of the bounding box along its sides.
        Parameters
        ----------
        all_sides : number, optional
            Value by which to extend the bounding box size along all
            sides.
        top : number, optional
            Value by which to extend the bounding box size along its top
            side.
        right : number, optional
            Value by which to extend the bounding box size along its right
            side.
        bottom : number, optional
            Value by which to extend the bounding box size along its bottom
            side.
        left : number, optional
            Value by which to extend the bounding box size along its left
            side.
        Returns
        -------
        green.utils.green_box.GBox
        Extended bounding box.
        """
        return self.deepcopy().extend_(all_sides, top, right, bottom, left)

    def intersection(self, other: 'GBox', default=None):
        """Compute the intersection BB between this BB and another BB.
        Note that in extreme cases, the intersection can be a single point.
        In that case the intersection bounding box exists and it will be
        returned, but it will have a height and width of zero.
        Parameters
        ----------
        other : green.utils.green_box.GBox
            Other bounding box with which to generate the intersection.
        default : any, optional
            Default value to return if there is no intersection.
        Returns
        -------
        green.utils.green_box.GBox or any
            Intersection bounding box of the two bounding boxes if there is
            an intersection.
            If there is no intersection, the default value will be returned,
            which can by anything.
        """
        if other.empty or self.empty:
            return GBox()
        x1_i = max(self.x1, other.x1)
        y1_i = max(self.y1, other.y1)
        x2_i = min(self.x2, other.x2)
        y2_i = min(self.y2, other.y2)
        if x1_i > x2_i or y1_i > y2_i:
            return default
        return GBox(x1=x1_i, y1=y1_i, x2=x2_i, y2=y2_i)

    def union(self, other):
        """Compute the union BB between this Box and another Box.
        This is equivalent to drawing a bounding box around all corner points
        of both bounding boxes.
        Parameters
        ----------
        other : green.utils.green_box.GBox
            Other bounding box with which to generate the union.
        Returns
        -------
        green.utils.green_box.GBox
            Union bounding box of the two bounding boxes.
        """
        if other.empty:
            return self
        if self.empty:
            return other
        return GBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def iou(self, other) -> float:
        """Compute the IoU between this bounding box and another one.
        IoU is the intersection over union, defined as::
            ``area(intersection(A, B)) / area(union(A, B))``
            ``= area(intersection(A, B))
                / (area(A) + area(B) - area(intersection(A, B)))``
        Parameters
        ----------
        other : green.utils.green_box.GBox
            Other bounding box with which to compare.
        Returns
        -------
        float
            IoU between the two bounding boxes.
        """
        inters = self.intersection(other)
        if inters is None:
            return 0.0
        area_union = self.area + other.area - inters.area
        return inters.area / area_union if area_union > 0 else 0.0

    def compute_out_of_image_area(self, image) -> float:
        """Compute the area of the BB that is outside of the image plane.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        float
            Total area of the bounding box that is outside of the image plane.
            Can be ``0.0``.
        """
        shape = normalize_imglike_shape(image)
        height, width = shape[0:2]
        bb_image = GBox(x1=0, y1=0, x2=width, y2=height)
        inter = self.intersection(bb_image, default=None)
        area = self.area
        return area if inter is None else area - inter.area

    def compute_out_of_image_fraction(self, image) -> float:
        """Compute fraction of BB area outside of the image plane.
        This estimates ``f = A_ooi / A``, where ``A_ooi`` is the area of the
        bounding box that is outside of the image plane, while ``A`` is the
        total area of the bounding box.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        float
            Fraction of the bounding box area that is outside of the image
            plane. Returns ``0.0`` if the bounding box is fully inside of
            the image plane. If the bounding box has an area of zero, the
            result is ``1.0`` if its coordinates are outside of the image
            plane, otherwise ``0.0``.
        """
        area = self.area
        if area == 0:
            shape = normalize_imglike_shape(image)
            height, width = shape[0:2]
            y1_outside = self.y1 < 0 or self.y1 >= height
            x1_outside = self.x1 < 0 or self.x1 >= width
            is_outside = (y1_outside or x1_outside)
            return 1.0 if is_outside else 0.0
        return self.compute_out_of_image_area(image) / area

    def is_fully_within_image(self, image) -> bool:
        """Estimate whether the bounding box is fully inside the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        bool
            ``True`` if the bounding box is fully inside the image area.
            ``False`` otherwise.
        """
        shape = normalize_imglike_shape(image)
        height, width = shape[0:2]
        return (
                self.x1 >= 0
                and self.x2 < width
                and self.y1 >= 0
                and self.y2 < height)

    def is_partly_within_image(self, image) -> bool:
        """Estimate whether the BB is at least partially inside the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape
            and must contain at least two integers.
        Returns
        -------
        bool
            ``True`` if the bounding box is at least partially inside the
            image area.
            ``False`` otherwise.
        """
        shape = normalize_imglike_shape(image)
        height, width = shape[0:2]
        eps = np.finfo(np.float32).eps
        img_bb = GBox(x1=0, x2=width - eps, y1=0, y2=height - eps)
        return self.intersection(img_bb) is not None

    def is_out_of_image(self, image, fully=True, partly=False) -> bool:
        """Estimate whether the BB is partially/fully outside of the image area.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two integers.
        fully : bool, optional
            Whether to return ``True`` if the bounding box is fully outside
            of the image area.
        partly : bool, optional
            Whether to return ``True`` if the bounding box is at least
            partially outside fo the image area.
        Returns
        -------
        bool
            ``True`` if the bounding box is partially/fully outside of the
            image area, depending on defined parameters.
            ``False`` otherwise.
        """
        if self.is_fully_within_image(image):
            return False
        if self.is_partly_within_image(image):
            return partly
        return fully

    def clip_out_of_image_(self, image):
        """Clip off parts of the BB box that are outside of the image in-place.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the bounding box.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two integers.
        Returns
        -------
        green.utils.green_box.GBox
            Bounding box, clipped to fall within the image dimensions.
            The object may have been modified in-place.
        """
        shape = normalize_imglike_shape(image)

        height, width = shape[0:2]
        assert height > 0, (
                "Expected image with height>0, got shape %s." % (image.shape,))
        assert width > 0, (
                "Expected image with width>0, got shape %s." % (image.shape,))

        eps = np.finfo(np.float32).eps
        self.x1 = np.clip(self.x1, 0, width - eps)
        self.x2 = np.clip(self.x2, 0, width - eps)
        self.y1 = np.clip(self.y1, 0, height - eps)
        self.y2 = np.clip(self.y2, 0, height - eps)

        return self

    def clip_out_of_image(self, image):
        """Clip off all parts of the BB box that are outside of the image.
        Parameters
        ----------
        image : (H,W,...) ndarray or tuple of int
            Image dimensions to use for the clipping of the bounding box.
            If an ``ndarray``, its shape will be used.
            If a ``tuple``, it is assumed to represent the image shape and
            must contain at least two integers.
        Returns
        -------
        green.utils.green_box.GBox
            Bounding box, clipped to fall within the image dimensions.
        """
        return self.deepcopy().clip_out_of_image_(image)

    def shift_(self, x=0, y=0):
        """Move this bounding box along the x/y-axis in-place.
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
        green.utils.green_box.GBox
            Shifted bounding box.
            The object may have been modified in-place.
        """
        self.x1 += x
        self.x2 += x
        self.y1 += y
        self.y2 += y
        return self

    def shift(self, x=0, y=0):
        """Move this bounding box along the x/y-axis.
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
        green.utils.green_box.GBox
            Shifted bounding box.
        """
        return self.deepcopy().shift_(x, y)


    def draw_on_image(self, image, color=np.random.randint(0, 255, 3), thickness=2, font_scale=0.4, show=False,
                      in_place=False):
        """Draw the bounding box on an image.
        This will automatically also draw the label, unless it is ``None``.

        Parameters
        ----------
        image : (H,W,C) ndarray
            The image onto which to draw the bounding box.
            Currently expected to be ``uint8``.
        color : iterable of int, optional
            The color to use, corresponding to the channel layout of the
            image. Usually RGB.
        thickness : int, optional
            The thickness of the bounding box in pixels.
        font_scale : float, optional
            The size of the text
        show : bool, optional, default is False
            if True plotting the image with matplotlib plt.show()
        Returns
        -------
        (H,W,C) ndarray(uint8)
            Image with bounding box drawn on it.
        Examples
        --------
        >> img = np.zeros(shape=(100,100,3))
        >> box1 = GBox(1,2,3,4, class_name='foo')
        >> draw_img = box1.draw_on_image(img, show=True)
        """

        if self.empty:
            return image
        if not in_place:
            image = np.copy(image)
        cv2.rectangle(img=image,
                      pt1=(self.x1_int, self.y1_int),
                      pt2=(self.x2_int, self.y2_int),
                      color=tuple(map(int, color)),
                      lineType=cv2.LINE_AA,
                      thickness=thickness)

        (retval, baseLine) = cv2.getTextSize(self.full_label, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=font_scale, thickness=1)
        textOrg = (self.x1_int, self.y1_int - 0)
        cv2.rectangle(image, (textOrg[0] - 0, textOrg[1] + baseLine - 0),
                      (textOrg[0] + retval[0] + 0, textOrg[1] - retval[1] - 0), (0, 0, 0), 2)
        cv2.rectangle(image, (textOrg[0] - 0, textOrg[1] + baseLine - 0),
                      (textOrg[0] + retval[0] + 0, textOrg[1] - retval[1] - 0), (255, 255, 255), -1)
        cv2.putText(image, self.full_label, textOrg, cv2.FONT_HERSHEY_DUPLEX, color=(0, 0, 0), fontScale=font_scale,
                    thickness=1)
        if show:
            plt.imshow(image)
            plt.show()
        return image


    def coords_almost_equals(self, other, max_distance=1e-4):
        """Estimate if this and another BB have almost identical coordinates.
        Parameters
        ----------
        other : green.utils.green_box.GBox or iterable
            The other bounding box with which to compare this one.
            If this is an ``iterable``, it is assumed to represent the top-left
            and bottom-right coordinates of that bounding box, given as e.g.
            an ``(2,2)`` ndarray or an ``(4,)`` ndarray or as a similar list.
        max_distance : number, optional
            The maximum euclidean distance between a corner on one bounding
            box and the closest corner on the other bounding box. If the
            distance is exceeded for any such pair, the two BBs are not
            viewed as equal.
        Returns
        -------
        bool
            Whether the two bounding boxes have almost identical corner
            coordinates.
        """
        if isinstance(other, GBox):
            coords_b = other.coords.flat
        elif is_np_array(other):
            # we use flat here in case other is (N,2) instead of (4,)
            coords_b = other.flat
        elif is_iterable(other):
            coords_b = list(flatten(other))
        else:
            raise ValueError(
                "Expected 'other' to be an iterable containing two "
                "(x,y)-coordinate pairs or a BoundingBox. "
                f"Got type {type(other)}.")

        coords_a = self.coords

        return np.allclose(coords_a.flat, coords_b, atol=max_distance, rtol=0)

    @classmethod
    def from_point_soup(cls, xy):
        """Convert a ``(2P,) or (P,2) ndarray`` to a BB instance.
        This is the inverse of
        :func:`~green_box.GBoxesOnImage.to_xyxy_array`.
        Parameters
        ----------
        xy : (2P,) ndarray or (P, 2) array or iterable of number or iterable of iterable of number
            Array containing ``P`` points in xy-form denoting a soup of
            points around which to place a bounding box.
            The array should usually be of dtype ``float32``.
        Returns
        -------
        green.utils.green_box.GBox
            Bounding box around the points.
        """
        xy = np.array(xy, dtype=np.float32)

        assert len(xy) > 0, (
                "Expected to get at least one point to place a bounding box "
                "around, got shape %s." % (xy.shape,))

        assert xy.ndim == 1 or (xy.ndim == 2 and xy.shape[-1] == 2), (
                "Expected input array of shape (P,) or (P, 2), "
                "got shape %s." % (xy.shape,))

        if xy.ndim == 1:
            xy = xy.reshape((-1, 2))

        x1, y1 = np.min(xy, axis=0)
        x2, y2 = np.max(xy, axis=0)

        return cls(x1=x1, y1=y1, x2=x2, y2=y2)

    @staticmethod
    def from_dict(box_dict:dict):
        """Creating GBox object from dict.
        When pickling Objects it's best to use python's builtins,
        So this method can be used to deserialize GBox.
        This is the inverse of
        :func:`~green.utils.green_box.GBox.to_dict`.
        Parameters
        ----------
        box_dict: dict
            must have the keys 'x1','x2','y1' and 'y2'
            The rest of the items will be in metadata.

        Returns
        -------
        green.utils.green_box.GBox
            created from  the box_dict.
        """
        box_dict = box_dict.copy()
        return GBox(
            x1=box_dict.pop('x1'),
            x2=box_dict.pop('x2'),
            y1=box_dict.pop('y1'),
            y2=box_dict.pop('y2'),
            class_name=box_dict.pop('class', None),
            metadata=box_dict)

    @staticmethod
    def from_imgaug_box(imgaug_box: imgaug.BoundingBox) -> 'GBox':
        """
        converts imgaug bounding box to green boxe

        Parameters
        ----------
         imgaug_boxes: list
            list of imgaug.BoundingBox objects
        Returns
        -------
        list of dicts
            each dict have the keys 'x1','x2', 'y1', 'y2', 'class', ... .
        """
        gbox = GBox(x1=imgaug_box.x1, y1=imgaug_box.y1, x2=imgaug_box.x2, y2=imgaug_box.y2, metadata=imgaug_box.label)
        return gbox

    def to_dict(self, cast_to_int: bool = False) -> dict:
        """Creating a Box-like Dict - a dict with keys 'x1','x2','y1' and 'y2' and metadata.
        When pickling Objects it's best to use python's builtins,
        So this method can be used to serialize GBox.
        This is the inverse of
        :func:`~green.utils.green_box.GBox.from_dict`.

        Returns
        -------
        dict
            created from  self, with keys 'x1','x2','y1' and 'y2' and additional metadata.
        """
        if cast_to_int:
            return {'x1': self.x1_int, 'y1': self.y1_int, 'x2': self.x2_int, 'y2': self.y2_int, 'class': self.class_name, **self.metadata}
        else:
            return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2, 'class': self.class_name, **self.metadata}

    def to_imgaug_box(self) -> imgaug.BoundingBox:
        """
        Creating imgaug package box from our box

        Returns
        -------
        imgaug.BoundingBox
        """
        to_label = self.metadata
        to_label['class'] = self.class_name
        return imgaug.BoundingBox(x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2, label=to_label)

    def copy(self, x1=None, y1=None, x2=None, y2=None, metadata=None, class_name=None) -> 'GBox':
        """Create a shallow copy of this BoundingBox instance.
        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.
        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.
        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.
        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.
        metadata : None or dict
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.
        Returns
        -------
        green.utils.green_box.GBox
            Shallow copy.
        """
        if metadata:
            class_from_meta = metadata.get('class')
            if class_from_meta:
                metadata = {k: v for k,v in metadata.items() if k !='class'}
            class_from_input = class_name or class_from_meta
        else:
            class_from_input = class_name


        return GBox(
            x1= x1 or self.x1,
            x2= x2 or self.x2,
            y1= y1 or self.y1,
            y2= y2 or self.y2,
            class_name=class_from_input or self.class_name,
            metadata=metadata or copy.deepcopy(self.metadata),
        )

    def deepcopy(self, x1=None, y1=None, x2=None, y2=None, metadata=None) -> 'GBox':
        """
        Create a deep copy of the BoundingBox object.
        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.
        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.
        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.
        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.
        metadata : None or dict
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.
        Returns
        -------
        green.utils.green_box.GBox
            Deep copy.
        """
        return self.copy(x1=x1, y1=y1, x2=x2, y2=y2, metadata=metadata)

    def get(self, key: str, default=None) -> any:
        """
        For supporting a dict like calls. for example ``gbox['x1']`` (in addition to gbox.x1)
        if the key is not one of ['class', 'x1', 'x2', 'y1', 'y2'], the @param default will be returned.

        Parameters
        ----------
         key : str
            the desired attribute
         default : any
            if the key is not one of the valid options, or if the key is valid, but the value was None, the @param
            default will be returned.
        Returns
        -------
            the value of the key, if the key is one of ['class', 'x1', 'x2', 'y1', 'y2'] and is not None, otherwise
            @param default will be returned.
        """
        value = None
        if key == 'class':
            value = self.class_name
        elif key == 'x1':
            value = self.x1
        elif key == 'x2':
            value = self.x2
        elif key == 'y1':
            value = self.y1
        elif key == 'y2':
            value = self.y2

        if value is None:
             value = default

        return value


    def __getitem__(self, item):
        """Get the coordinate(s) with given indices.
        Returns
        -------
        ndarray
            xy-coordinate(s) as ``ndarray``.
        """
        if item == 'class':
            return self.class_name
        if item == 'x1':
            return self.x1
        if item == 'x2':
            return self.x2
        if item == 'y1':
            return self.y1
        if item == 'y2':
            return self.y2
        if isinstance(item, (int,slice)):
            return self.coords[item]

    def __setitem__(self, key, value):
        if key == 'class':
            self.class_name = value
        if key == 'x1':
            self.x1 = value
        if key == 'x2':
            self.x2 = value
        if key == 'y1':
            self.y1 = value
        if key == 'y2':
            self.y2 = value

    def __hash__(self) -> int:
        """
        for creating sets of GBoxes

        Returns
        -------
        number
            the hash code of the box
        Examples
        --------
        >>  b1 = GBox(1, 2, 3, 4, class_name='foo')
        >>  b2 = GBox(1, 2, 3, 4, class_name='foo')
        >>  b3 = GBox(1, 2, 3, 4, metadata={'class': 'foo', 'blabla': 'blu'})

        >> {b1, b2}   # --> {GBox(1, 2, 3, 4, class_name='foo')}
        >> {b1, b3}   # --> {GBox(1, 2, 3, 4, class_name='foo'), GBox(1, 2, 3, 4, metadata={'class': 'foo', 'blabla': 'blu'})}
        >> {b1,b2,b3} # --> {GBox(1, 2, 3, 4, class_name='foo'), GBox(1, 2, 3, 4, metadata={'class': 'foo', 'blabla': 'blu'})}

        """

        return hash((self.x1, self.y1, self.x2, self.y2, self.class_name, str(self.metadata)))

    def __iter__(self):
        """Iterate over the coordinates of this instance.
        Yields
        ------
        ndarray
            An ``(2,)`` ``ndarray`` denoting an xy-coordinate pair.
        """
        return iter(self.coords)

    def __eq__(self, other: 'GBox') -> bool:
        """
        Determines if 2 boxes are equal.
        2 boxes are equal iff they have same coordinates, class name and metadata.

        Parameters
        ----------
        other : GBox
            the other box to check if equal to self
        Returns
        -------
        bool
            True if the coordinates, class_name and metadata of the 2 boxes are equal
            False otherwise

        Examples
        --------
        >> b1 = GBox(1,2,3,4, class_name='foo')
        >> b2 = GBox(1,2,3,4, metadata={'class':'foo'})
        >> b3 = GBox(1,2,3,4, class_name='not foo')
        >> b4 = GBox(10,2,3,4, class_name='foo')

        >> b1 == b2 # -> True (different ways to init)
        >> b1 == b3 # -> False (different class names)
        >> b1 == b4 # -> False (different x1)

        """
        if isinstance(other, GBox):
            this_meta_without_class  = {k:v for k,v in self.metadata.items() if k != 'class'}
            other_meta_without_class = {k:v for k,v in other.metadata.items() if k != 'class'}
            return (self.x1 == other.x1 and
                    self.x2 == other.x2 and
                    self.y1 == other.y1 and
                    self.y2 == other.y2 and
                    self.class_name == other.class_name and
                    this_meta_without_class == other_meta_without_class)
        else:
            return False

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"GBox(x1={self.x1:.4f}, y1={self.y1:.4f}, x2={self.x2:.4f}, y2={self.y2:.4f}, label={self.full_label})"






class PredGBox(GBox):

    def __init__(
            self,
            x1: float = 0,
            y1: float = 0,
            x2: float = 0,
            y2: float = 0,
            probs: Union[float, OrderedDict] = None,
            metadata: dict = None,
            class_name: str = None,
            extras: Any = None,
    ):
        super().__init__(x1, y1, x2, y2, metadata, class_name)
        if isinstance(probs, dict):
            probs = OrderedDict(probs)
        elif not isinstance(probs, (float, int)):
            logger.warning(f"PredGBox got bad type of probs: {type(probs)}, {probs}, when expecting to dict or float")
            probs = -1

        self.probs  : Union[float, OrderedDict] = probs
        self.extras : Any                       = extras

    @property
    def label(self) -> str:
        """The label to be shown in printings, plotting etc.

        Returns
        -------
        str
            the class name of the box if it's not None else empty string ""
        """
        class_label = self.class_name
        if class_label:
            return f"{class_label}: {self.top_prob}"
        else:
            return ""



    @property
    def full_label(self) -> str:
        """The full label of the box constructed from the effective class name and the class name.
        for example if the effective class name is `Broad Leaf 1-2` , and the class name is 'Scientific name`
        the result will be `Broad Leaf 1-2(Scientific name)`
        if there isn't class name the result will be the self.class_name.
        """

        return self.label

    @property
    def top_prob(self) -> float:
        if isinstance(self.probs, dict):
            try:
                return next(iter(self.probs))
            except:
                return -1
        else: # already a number
            return self.probs



    @staticmethod
    def from_dict(box_dict: dict) -> 'PredGBox':
        """Creating PredGBox object from dict.
        When pickling Objects it's best to use python's builtins,
        So this method can be used to deserialize PredGBox.
        This is the inverse of
        :func:`~green.utils.green_box.PredGBox.to_dict`.
        Parameters
        ----------
        box_dict: dict
            must have the keys 'x1','x2','y1','y2' and probs
            The rest of the items will be in metadata.

        Returns
        -------
        green.utils.green_box.PredGBox
            created from  the box_dict.
        """
        return PredGBox(
            x1=box_dict.pop('x1'),
            x2=box_dict.pop('x2'),
            y1=box_dict.pop('y1'),
            y2=box_dict.pop('y2'),
            probs=box_dict.pop('probs'),
            class_name=box_dict.pop('class', None),
            metadata=box_dict)

    def to_dict(self, cast_to_int: bool = False) -> dict:
        """Creating a Box-like Dict - a dict with keys 'x1','x2','y1','y2', probs and metadata.
        When pickling Objects it's best to use python's builtins,
        So this method can be used to serialize GBox.
        This is the inverse of
        :func:`~green.utils.green_box.PredGBox.from_dict`.

        Returns
        -------
        dict
            created from  self, with keys 'x1','x2','y1','y2', 'probs', and additional metadata.
        """
        if cast_to_int:
            return {'x1': self.x1_int, 'y1': self.y1_int, 'x2': self.x2_int, 'y2': self.y2_int, 'probs': self.probs,
                    'class': self.class_name, **self.metadata}

        else:
            return {'x1': self.x1, 'y1': self.y1, 'x2': self.x2, 'y2': self.y2, 'probs': self.probs,
                    'class': self.class_name, **self.metadata}

    def __eq__(self, other: 'GBox') -> bool:
        """
        Determines if 2 boxes are equal.
        2 boxes are equal iff they have same coordinates, class name and metadata.

        Parameters
        ----------
        other : GBox
            the other box to check if equal to self
        Returns
        -------
        bool
            True if the coordinates, class_name and metadata of the 2 boxes are equal
            False otherwise

        Examples
        --------
        >> b1 = GBox(1,2,3,4, class_name='foo')
        >> b2 = GBox(1,2,3,4, metadata={'class':'foo'})
        >> b3 = GBox(1,2,3,4, class_name='not foo')
        >> b4 = GBox(10,2,3,4, class_name='foo')

        >> b1 == b2 # -> True (different ways to init)
        >> b1 == b3 # -> False (different class names)
        >> b1 == b4 # -> False (different x1)

        """
        if isinstance(other, PredGBox):
            this_meta_without_class  = {k:v for k,v in self.metadata.items() if k != 'class'}
            other_meta_without_class = {k:v for k,v in other.metadata.items() if k != 'class'}
            return (self.x1 == other.x1 and
                    self.x2 == other.x2 and
                    self.y1 == other.y1 and
                    self.y2 == other.y2 and
                    self.class_name == other.class_name and
                    self.probs == other.probs and
                    this_meta_without_class == other_meta_without_class)
        else:
            return False


    def __hash__(self) -> int:
        """
        for creating sets of GBoxes

        Returns
        -------
        number
            the hash code of the box

        """

        return hash((self.x1, self.y1, self.x2, self.y2, self.class_name, str(self.metadata), str(self.probs)))

    def copy(self, x1=None, y1=None, x2=None, y2=None, metadata=None, class_name=None, probs=None) -> 'GBox':
        """Create a shallow copy of this BoundingBox instance.
        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.
        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.
        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.
        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.
        metadata : None or dict
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.
        Returns
        -------
        green.utils.green_box.GBox
            Shallow copy.
        """
        if metadata:
            class_from_meta = metadata.get('class')
            if class_from_meta:
                metadata = {k: v for k,v in metadata.items() if k !='class'}
            class_from_input = class_name or class_from_meta
        else:
            class_from_input = class_name


        return PredGBox(
            x1= x1 or self.x1,
            x2= x2 or self.x2,
            y1= y1 or self.y1,
            y2= y2 or self.y2,
            probs= probs or self.probs,
            class_name=class_from_input or self.class_name,
            metadata=metadata or copy.deepcopy(self.metadata),
        )

    def deepcopy(self, x1=None, y1=None, x2=None, y2=None, metadata=None, probs=None) -> 'GBox':
        """
        Create a deep copy of the BoundingBox object.
        Parameters
        ----------
        x1 : None or number
            If not ``None``, then the ``x1`` coordinate of the copied object
            will be set to this value.
        y1 : None or number
            If not ``None``, then the ``y1`` coordinate of the copied object
            will be set to this value.
        x2 : None or number
            If not ``None``, then the ``x2`` coordinate of the copied object
            will be set to this value.
        y2 : None or number
            If not ``None``, then the ``y2`` coordinate of the copied object
            will be set to this value.
        metadata : None or dict
            If not ``None``, then the ``label`` of the copied object
            will be set to this value.
        Returns
        -------
        green.utils.green_box.GBox
            Deep copy.
        """
        return self.copy(x1=x1, y1=y1, x2=x2, y2=y2, metadata=metadata, probs=probs)

    def get(self, key: str, default=None) -> any:
        """
        For supporting a dict like calls. for example ``gbox['x1']`` (in addition to gbox.x1)
        if the key is not one of ['class', 'x1', 'x2', 'y1', 'y2'], the @param default will be returned.

        Parameters
        ----------
         key : str
            the desired attribute
         default : any
            if the key is not one of the valid options, or if the key is valid, but the value was None, the @param
            default will be returned.
        Returns
        -------
            the value of the key, if the key is one of ['class', 'x1', 'x2', 'y1', 'y2'] and is not None, otherwise
            @param default will be returned.
        """
        value = None
        if key == 'class':
            value = self.class_name
        elif key == 'x1':
            value = self.x1
        elif key == 'x2':
            value = self.x2
        elif key == 'y1':
            value = self.y1
        elif key == 'y2':
            value = self.y2
        elif key == 'probs':
            value = self.probs

        if value is None:
             value = default

        return value


    def __getitem__(self, item):
        """Get the coordinate(s) with given indices.
        Returns
        -------
        ndarray
            xy-coordinate(s) as ``ndarray``.
        """
        if item == 'class':
            return self.class_name
        if item == 'x1':
            return self.x1
        if item == 'x2':
            return self.x2
        if item == 'y1':
            return self.y1
        if item == 'y2':
            return self.y2
        if item == 'probs':
            return self.probs
        if isinstance(item, (int,slice)):
            return self.coords[item]