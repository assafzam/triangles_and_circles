from numbers import Number

import logging
logger = logging.getLogger()
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from typing import Union, List, Tuple

import numpy as np

def project_coords(coords: Union[np.ndarray, List[Tuple[float, float]]],
                   from_shape: Union[Tuple[int], np.ndarray],
                   to_shape: Union[Tuple[int], np.ndarray]
                   ) -> np.ndarray:
    """Project coordinates from one image shape to another.
    This performs a relative projection, e.g. a point at ``60%`` of the old
    image width will be at ``60%`` of the new image width after projection.
    Parameters
    ----------
    coords : ndarray or list of tuple of number
        Coordinates to project.
        Either an ``(N,2)`` numpy array or a ``list`` containing ``(x,y)``
        coordinate ``tuple`` s.
    from_shape : tuple of int or ndarray
        Old image shape.
    to_shape : tuple of int or ndarray
        New image shape.
    Returns
    -------
    ndarray
        Projected coordinates as ``(N,2)`` ``float32`` numpy array.
    """
    if is_np_array(coords):
        coords = np.copy(coords)

    return project_coords_(coords, from_shape, to_shape)


def project_coords_(coords: Union[np.ndarray, List[Tuple[float, float]]],
                    from_shape: Union[Tuple[int], np.ndarray],
                    to_shape: Union[Tuple[int], np.ndarray]
                    ) -> np.ndarray:
    """Project coordinates from one image shape to another in-place.
    This performs a relative projection, e.g. a point at ``60%`` of the old
    image width will be at ``60%`` of the new image width after projection.
    Added in 0.4.0.
    Parameters
    ----------
    coords : ndarray or list of tuple of number
        Coordinates to project.
        Either an ``(N,2)`` numpy array or a ``list`` containing ``(x,y)``
        coordinate ``tuple`` s.
    from_shape : tuple of int or ndarray
        Old image shape.
    to_shape : tuple of int or ndarray
        New image shape.
    Returns
    -------
    ndarray
        Projected coordinates as ``(N,2)`` ``float32`` numpy array.
        This function may change the input data in-place.
    """
    from_shape = normalize_shape(from_shape)
    to_shape = normalize_shape(to_shape)
    if from_shape[0:2] == to_shape[0:2]:
        return coords

    from_height, from_width = from_shape[0:2]
    to_height, to_width = to_shape[0:2]

    no_zeros_in_shapes = (
        all([v > 0 for v in [from_height, from_width, to_height, to_width]]))
    assert no_zeros_in_shapes, f"Expected from_shape and to_shape to not contain zeros. Got {from_shape} and {to_shape}"

    coords_proj = coords
    if not is_np_array(coords) or coords.dtype.kind != "f":
        coords_proj = np.array(coords).astype(np.float32)

    coords_proj[:, 0] = (coords_proj[:, 0] / from_width) * to_width
    coords_proj[:, 1] = (coords_proj[:, 1] / from_height) * to_height

    return coords_proj


def normalize_shape(shape: Union[Tuple[int], np.ndarray]) -> Tuple[int]:
    """Normalize a shape ``tuple`` or ``array`` to a shape ``tuple``.
    Parameters
    ----------
    shape : tuple of int or ndarray
        The input to normalize. May optionally be an array.
    Returns
    -------
    tuple of int
        Shape ``tuple``.
    """
    if isinstance(shape, tuple):
        return shape
    assert is_np_array(shape), f"Expected tuple of ints or array, got {type(shape)}."
    return shape.shape


def normalize_imglike_shape(shape: Union[tuple, np.ndarray]) -> tuple:
    """Normalize a shape tuple or image-like ``array`` to a shape tuple.
    Parameters
    ----------
    shape : tuple of int or ndarray
        The input to normalize. May optionally be an array. If it is an
        array, it must be 2-dimensional (height, width) or 3-dimensional
        (height, width, channels). Otherwise an error will be raised.
    Returns
    -------
    tuple of int
        Shape ``tuple``.
    """
    if isinstance(shape, tuple):
        return shape
    assert is_np_array(shape), f"Expected tuple of ints or array, got {type(shape)}"
    shape = shape.shape
    assert len(shape) in [2, 3], (
        f"Expected image array to be 2-dimensional or 3-dimensional, got {len(shape)}-dimensional input of shape {shape}"
    )
    return shape


def flatten(nested_iterable):
    """Flatten arbitrarily nested lists/tuples.
    Code partially taken from https://stackoverflow.com/a/10824420.
    Parameters
    ----------
    nested_iterable
        A ``list`` or ``tuple`` of arbitrarily nested values.
    Yields
    ------
    any
        All values in `nested_iterable`, flattened.
    """
    # don't just check if something is iterable here, because then strings
    # and arrays will be split into their characters and components
    if not isinstance(nested_iterable, (list, tuple)):
        yield nested_iterable
    else:
        for i in nested_iterable:
            if isinstance(i, (list, tuple)):
                for j in flatten(i):
                    yield j
            else:
                yield i


def _remove_out_of_image_fraction_(gboxes_on_img, fraction):
    gboxes_on_img.items = [
        item for item in gboxes_on_img.items
        if item.compute_out_of_image_fraction(gboxes_on_img.img_shape) < fraction]
    return gboxes_on_img


def is_np_array(val: any) -> bool:
    """Check whether a variable is a numpy array.
    Parameters
    ----------
    val
        The variable to check.
    Returns
    -------
    bool
        ``True`` if the variable is a numpy array. Otherwise ``False``.
    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic))
    # seems to also fire for scalar numpy values even though those are not
    # arrays
    return isinstance(val, np.ndarray)


def is_iterable(val):
    """
    Checks whether a variable is iterable.
    Parameters
    ----------
    val
        The variable to check.
    Returns
    -------
    bool
        ``True`` if the variable is an iterable. Otherwise ``False``.
    """
    return isinstance(val, Iterable)


def convert_pixels_2_meter(pixels: Union[float, int], mm_per_pixel: Union[float, int]) -> float:
    return mm_per_pixel * pixels / 1000

def is_valid_box_dict(d):
    return (isinstance(d, dict)             and
            isinstance(d.get('x1'), Number) and
            isinstance(d.get('y1'), Number) and
            isinstance(d.get('x2'), Number) and
            isinstance(d.get('y2'), Number)
            )

def is_valid_pred_box_dict(d):
    return is_valid_box_dict(d) and isinstance(d.get('probs'), (dict, float, int))

def verify_gbox_params(args):
    x1         = args.get('x1', None)
    x2         = args.get('x2', None)
    y1         = args.get('y1', None)
    y2         = args.get('y2', None)
    class_name = args.get('class_name', None)
    metadata   = args.get('metadata', None)

    # gof types:
    assert isinstance(x1, (int, float)) or np.isreal(x1), f"GBox coordinates must be integers or floats (Got {x1}, type:{type(x1)})"
    assert isinstance(x2, (int, float)) or np.isreal(x2), f"GBox coordinates must be integers or floats (Got {x2}, type:{type(x2)})"
    assert isinstance(y1, (int, float)) or np.isreal(y1), f"GBox coordinates must be integers or floats (Got {y1}, type:{type(y1)})"
    assert isinstance(y2, (int, float)) or np.isreal(y2), f"GBox coordinates must be integers or floats (Got {y2}, type:{type(y2)})"
    assert isinstance(metadata, dict) or metadata is None, f"meta data of GBox must be a dict! (Got {metadata}, type: {type(metadata)}) "
    assert isinstance(class_name, str) or class_name is None, f"class_name of GBox must be str! (Got {class_name}, type: {type(class_name)})"

    # class name may be in metadata also
    if class_name and isinstance(metadata,dict) and metadata.get('class') and (class_name != metadata.get('class')):
        logger.warning(f"Got class_name: {class_name}, and metadata with class: {metadata.get('class')}, Not Equal!, using {class_name}")

def get_ratio_fix(o_width, o_height, resized_width, resized_height):
    height_ratio = o_height / (resized_height * 1.0)
    width_ratio = o_width / (resized_width * 1.0)
    return height_ratio, width_ratio