import numpy as np
import skimage.transform
from pathlib import Path
import tqdm as tqdm
import tensorflow as tf
import h5netcdf
from itertools import product
import data_utils
import hashlib

BATCH_SIZE = 16
PATCH_SIZE = 256

def build_dataset(name, method_name='zoomnn'):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    all_npzs = list(Path('data_np').glob('*.npz'))

    filter_fn_raw = lambda path: int(hashlib.md5(path.stem.encode('utf-8')).hexdigest()[:2], 16) > 50
    if name == 'train':
        filter_fn = filter_fn_raw
    elif name == 'val':
        filter_fn = lambda x: not(filter_fn_raw(x))
    else:
        raise ValueError(f"Can't build dataset called '{name}")
    np_files = [str(path) for path in all_npzs if filter_fn(path)]
    print(name, len(np_files))

    data = tf.data.Dataset.from_tensor_slices(np_files)
    data = data.shuffle(len(np_files))
    data = data.map(load_tile, num_parallel_calls=None, deterministic=False)
    if method_name == 'zoomnn':
        data = data.map(inflate_and_patch_zoomnn, num_parallel_calls=3, deterministic=False)
    elif method_name == 'unet':
        data = data.map(inflate_and_patch_unet, num_parallel_calls=3, deterministic=False)
    else:
        raise ValueError(f"Can't prepare dataset for {method_name}")
    data = data.interleave(tensors_to_dataset, cycle_length=3, num_parallel_calls=3)
    data = data.filter(filter_nodata)

    if name == 'train':
        data = data.shuffle(1024 * 8)
    #     data = data.repeat(64)
    #     data = data.map(augment)
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(12)
    return data


def load_tile(tile_path):
    def inner(tile_path):
        # For some reason, TF turns our nice strings into byte strings...
        tile_path = Path(tile_path.decode('utf-8'))
        npz = np.load(tile_path)

        sar = npz['nersc_sar']
        btemp = npz['btemp']
        labels = npz['concentration'][...,[0]]
        labels = np.where(labels == -1, -1, (labels >= 3).astype(np.int8))

        return sar, btemp, labels, sar.shape, btemp.shape, labels.shape
    sar, btemp, labels, sar_shp, btemp_shp, labels_shp = tf.numpy_function(inner,
            [tile_path], [tf.float32, tf.float32, tf.int8, tf.int64, tf.int64, tf.int64])

    sar = tf.reshape(sar, (sar_shp[0], sar_shp[1], sar_shp[2]))
    btemp = tf.reshape(btemp, (btemp_shp[0], btemp_shp[1], btemp_shp[2]))
    labels = tf.reshape(labels, (labels_shp[0], labels_shp[1], labels_shp[2]))

    return sar, btemp, labels


@tf.function
def inflate_and_patch_unet(sar, btemp, lbl):
    shp   = tf.shape(sar)[:2]
    AREA = tf.image.ResizeMethod.AREA
    BILINEAR = tf.image.ResizeMethod.BILINEAR

    sar_1 = sar
    btemp_4 = tf.image.resize(btemp, shp //  8, method=BILINEAR)

    s1 = sar_1
    s4 = btemp_4

    STR = PATCH_SIZE
    s1_patch = to_patches(s1, STR)
    lbl_patch = to_patches(lbl, STR)
    s4_patch = to_patches(s4, STR // 8, STR // 8)

    return s1_patch, s4_patch, lbl_patch


@tf.function
def inflate_and_patch_zoomnn(sar, btemp, lbl):
    shp   = tf.shape(sar)[:2]
    AREA = tf.image.ResizeMethod.AREA
    BILINEAR = tf.image.ResizeMethod.BILINEAR

    # TODO: Fix 'images' contains no shape (sar.shape == <unknown>)
    sar_1 = sar
    sar_2 = tf.image.resize(sar_1, shp //  2, method=AREA)
    sar_3 = tf.image.resize(sar_2, shp //  4, method=AREA)
    sar_4 = tf.image.resize(sar_3, shp //  8, method=AREA)

    btemp_4 = tf.image.resize(btemp, shp //  8, method=BILINEAR)

    s1 = sar_1
    s2 = sar_2
    s3 = sar_3
    s4 = tf.concat([sar_4, btemp_4], axis=-1)

    PS = PATCH_SIZE
    p2 = (PS - PS // 2) // 2
    s2 = tf.pad(s2, ((p2, p2), (p2, p2), (0, 0)), mode='REFLECT')

    p3 = (PS - PS // 4) // 2
    s3 = tf.pad(s3, ((p3, p3), (p3, p3), (0, 0)), mode='REFLECT')

    p4 = (PS - PS // 8) // 2
    s4 = tf.pad(s4, ((p4, p4), (p4, p4), (0, 0)), mode='REFLECT')

    STR = PATCH_SIZE
    s1_patch = to_patches(s1, STR)
    lbl_patch = to_patches(lbl, STR)

    s2_patch = to_patches(s2, STR // 2)
    s3_patch = to_patches(s3, STR // 4)
    s4_patch = to_patches(s4, STR // 8)

    return s1_patch, s2_patch, s3_patch, s4_patch, lbl_patch


def tensors_to_dataset(*tensors):
    return tf.data.Dataset.from_tensor_slices((tuple(tensors[:-1]), tensors[-1]))


def filter_nodata(img, lbl):
    s1 = img[0]
    nodata_pct = tf.reduce_mean(tf.cast(tf.math.logical_or(
            tf.squeeze(tf.math.equal(lbl, tf.constant(-1, tf.int8)), 2),
            tf.reduce_all(tf.equal(s1, tf.constant(0, tf.float32)), axis=-1)
    ), tf.float32))

    return tf.math.less(nodata_pct, 0.1)


def to_patches(img, strides, patch_size=PATCH_SIZE):
    channels = tf.shape(img)[2] 
    img = tf.expand_dims(img, 0)
    patches = tf.image.extract_patches(img,
        [1, patch_size, patch_size, 1],
        strides=[1, strides, strides, 1],
        rates=[1, 1, 1, 1], padding='VALID'
    )[0]
    return tf.reshape(patches, (-1, patch_size, patch_size, channels))


@tf.function
def crop_to_patchsize(image, cy, cx):
    HALF = tf.constant(PATCH_SIZE // 2)
    ZERO = tf.constant(0)
    H = tf.shape(image)[0]
    W = tf.shape(image)[1]

    y0 = tf.math.maximum(cy - HALF, ZERO)
    y1 = tf.math.minimum(cy + HALF, H)
    x0 = tf.math.maximum(cx - HALF, ZERO)
    x1 = tf.math.minimum(cx + HALF, W)

    p_y0 = tf.math.maximum(0, HALF - cy)
    p_y1 = tf.math.maximum(0, HALF - (H - cy))
    p_x0 = tf.math.maximum(0, HALF - cx)
    p_x1 = tf.math.maximum(0, HALF - (W - cx))

    crop = image[y0:y1, x0:x1]
    if p_y0 > 0 or p_y1 > 0 or p_x0 > 0 or p_x1 > 0:
        crop = tf.pad(crop, ((p_y0, p_y1), (p_x0, p_x1), (0, 0)), mode='REFLECT')

    return crop

