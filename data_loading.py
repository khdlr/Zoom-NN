import numpy as np
from pathlib import Path
import tqdm as tqdm
import tensorflow as tf
import netCDF4
from itertools import product
import data_utils
import hashlib

BATCH_SIZE = 16
PATCH_SIZE = 256

def build_dataset(name):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    all_ncs = Path('data').glob('*.nc')

    filter_fn_raw = lambda path: int(hashlib.md5(path.stem.encode('utf-8')).hexdigest()[:2], 16) > 50
    if name == 'train':
        filter_fn = filter_fn_raw
    elif name == 'val':
        filter_fn = lambda x: not(filter_fn_raw(x))
    else:
        raise ValueError(f"Can't build dataset called '{name}")
    nc_files = [str(path) for path in all_ncs if filter_fn(path)]

    data = tf.data.Dataset.from_tensor_slices(nc_files)
    data = data.map(lambda x: tf.numpy_function(load_tile, [x],
        [tf.float32, tf.float32, tf.int8]),
        num_parallel_calls=AUTOTUNE,
    )
    data = data.interleave(flow_from_tensors,
            cycle_length=3, deterministic=False,
            num_parallel_calls=AUTOTUNE
    )
    if name == 'train':
        data = data.shuffle(512)
    #     data = data.repeat(64)
    #     data = data.map(augment)
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(AUTOTUNE)
    return data


def load_tile(tile_path):
    # For some reason, TF turns our nice strings into byte strings...
    tile_path = tile_path.decode('utf-8')
    ncf = netCDF4.Dataset(tile_path)

    freqs = [6.9, 7.3, 10.7, 18.7, 23.8, 36.5, 89.0]
    polarizations = ['v', 'h']
    btemp_names = [f'btemp_{freq}{pol}' for pol, freq in product(polarizations, freqs)]

    sar    = data_utils.get_stack(ncf, ['sar_primary', 'sar_secondary'])
    btemp  = data_utils.get_stack(ncf, btemp_names).astype(np.float32) / 200
    labels = data_utils.get_labels(ncf).astype(np.int8)

    sar = np.nan_to_num(sar)
    btemp = np.nan_to_num(btemp)

    return sar, btemp, labels


def flow_from_tensors(sar, btemp, labels):
    slices = tf.py_function(
        flow_from_tensors_np,
        [sar, btemp, labels],
        [tf.float32] * 7 + [tf.int8]
    )
    return tf.data.Dataset.from_tensor_slices((dict(
        input_s1=slices[0],
        input_s2=slices[1],
        input_s3=slices[2],
        input_s4=slices[3],
        input_s5=slices[4],
        input_b4=slices[5],
        input_b5=slices[6],
        ), slices[-1]
    ))


def flow_from_tensors_np(sar, btemp, labels):
    shp   = tf.shape(sar)[:2]
    AREA = tf.image.ResizeMethod.AREA
    BILINEAR = tf.image.ResizeMethod.BILINEAR

    sar_1 = sar
    sar_2 = tf.image.resize(sar_1, shp //  2, method=AREA)
    sar_3 = tf.image.resize(sar_2, shp //  4, method=AREA)
    sar_4 = tf.image.resize(sar_3, shp //  8, method=AREA)
    sar_5 = tf.image.resize(sar_4, shp // 16, method=AREA)

    btemp_4 = tf.image.resize(btemp, shp //  8, method=BILINEAR)
    btemp_5 = tf.image.resize(btemp, shp // 16, method=BILINEAR)

    yvals = np.arange(0, sar_1.shape[0] - PATCH_SIZE, PATCH_SIZE)
    xvals = np.arange(0, sar_1.shape[1] - PATCH_SIZE, PATCH_SIZE)

    s1_slices = []
    s2_slices = []
    s3_slices = []
    s4_slices = []
    s5_slices = []

    b4_slices = []
    b5_slices = []

    label_slices = []

    HALF = PATCH_SIZE // 2

    for y0 in yvals:
        y1 = y0 + PATCH_SIZE
        for x0 in xvals:
            x1 = x0 + PATCH_SIZE

            cy = y0 + HALF
            cx = x0 + HALF

            label = labels[y0:y1, x0:x1]
            s1 = sar_1[y0:y1, x0:x1]

            nodata_pct = tf.reduce_mean(tf.cast(
                (label == -1) | tf.reduce_all(s1 == 0, axis=-1),
            tf.float32))
            if nodata_pct > 0.1:
                continue

            s1_slices.append(s1)
            label_slices.append(label)

            cy //= 2; cx //= 2
            s2 = crop_to_patchsize(sar_2, cy, cx)
            s2_slices.append(s2)

            cy //= 2; cx //= 2
            s3 = crop_to_patchsize(sar_3, cy, cx)
            s3_slices.append(s3)

            cy //= 2; cx //= 2
            s4 = crop_to_patchsize(sar_4, cy, cx)
            b4 = crop_to_patchsize(btemp_4, cy, cx)
            s4_slices.append(s4)
            b4_slices.append(b4)

            cy //= 2; cx //= 2
            s5 = crop_to_patchsize(sar_5, cy, cx)
            b5 = crop_to_patchsize(btemp_5, cy, cx)
            s5_slices.append(s5)
            b5_slices.append(b5)

    return tf.stack(s1_slices), \
         tf.stack(s2_slices), \
         tf.stack(s3_slices), \
         tf.stack(s4_slices), \
         tf.stack(s5_slices), \
         tf.stack(b4_slices), \
         tf.stack(b5_slices), \
         tf.stack(label_slices)


def crop_to_patchsize(image, cy, cx):
    HALF = PATCH_SIZE // 2
    y0 = max(cy-HALF, 0)
    y1 = min(cy+HALF, image.shape[0])
    x0 = max(cx-HALF, 0)
    x1 = min(cx+HALF, image.shape[1])

    p_y0 = max(0, HALF - cy)
    p_y1 = max(0, HALF - (image.shape[0] - cy))
    p_x0 = max(0, HALF - cx)
    p_x1 = max(0, HALF - (image.shape[1] - cx))

    crop = image[y0:y1, x0:x1]
    if p_y0 > 0 or p_y1 > 0 or p_x0 > 0 or p_x1 > 0:
        crop = tf.pad(crop, ((p_y0, p_y1), (p_x0, p_x1), (0, 0)), mode='REFLECT')

    return crop

