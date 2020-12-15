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

def build_dataset(name):
    global prog_bar

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
    data = data.map(lambda x: tf.numpy_function(load_tile, [x],
        [tf.float32, tf.bool]), num_parallel_calls=3,
    )

    data = data.prefetch(8)
    data = data.interleave(flow_from_tensors,
            cycle_length=3, deterministic=False, num_parallel_calls=3
    )
    if name == 'train':
        data = data.shuffle(2048)
    #     data = data.repeat(64)
    #     data = data.map(augment)
    data = data.batch(BATCH_SIZE)
    data = data.prefetch(AUTOTUNE)
    return data


def load_tile(tile_path):
    # For some reason, TF turns our nice strings into byte strings...
    tile_path = Path(tile_path.decode('utf-8'))
    npz = np.load(tile_path)

    sar = npz['sar']
    # btemp = npz['btemp']
    labels = npz['concentration'][..., 0] >= 10

    return sar, labels


def flow_from_tensors(sar, labels):
    return tf.data.Dataset.from_generator(flow_from_tensors_np,
        # ((tf.float32,) * 4, tf.bool),
        (tf.float32, tf.bool),
        args=[sar, labels],
    )


def flow_from_tensors_np(sar, labels):
    H, W = sar.shape[:2]

    print('=== Flow ===')

    sar_1 = sar
    sar_2 = skimage.transform.downscale_local_mean(sar_1, (2, 2, 1))
    sar_3 = skimage.transform.downscale_local_mean(sar_2, (2, 2, 1))
    sar_4 = skimage.transform.downscale_local_mean(sar_3, (2, 2, 1))

    # btemp_3 = skimage.transform.resize(btemp, (H // 4, W // 4))
    # btemp_4 = skimage.transform.resize(btemp, (H // 8, W // 8))

    yvals = np.arange(0, sar_1.shape[0] - PATCH_SIZE, PATCH_SIZE // 4)
    xvals = np.arange(0, sar_1.shape[1] - PATCH_SIZE, PATCH_SIZE // 4)

    HALF = PATCH_SIZE // 2

    yields = 0
    for y0 in yvals:
        y1 = y0 + PATCH_SIZE
        for x0 in xvals:
            x1 = x0 + PATCH_SIZE

            cy = y0 + HALF
            cx = x0 + HALF

            label = labels[y0:y1, x0:x1]
            s1 = sar_1[y0:y1, x0:x1]

            nodata_pct = np.mean((label == -1) | np.all(s1 == 0, axis=-1))
            if nodata_pct > 0.1:
                continue

            # cy //= 2; cx //= 2
            # s2 = crop_to_patchsize(sar_2, cy, cx)

            # cy //= 2; cx //= 2
            # s3 = crop_to_patchsize(sar_3, cy, cx)
            # b3 = crop_to_patchsize(btemp_3, cy, cx)

            # cy //= 2; cx //= 2
            # s4 = crop_to_patchsize(sar_4, cy, cx)
            # b4 = crop_to_patchsize(btemp_4, cy, cx)

            # yield ((
            #     s1,
            #     s2,
            #     tf.concat([s3, b3], axis=-1),
            #     tf.concat([s4, b4], axis=-1),
            # ), label)
            yield (s1, label)
            yields += 1
    print(f'Flow yielded {yields} times')


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

