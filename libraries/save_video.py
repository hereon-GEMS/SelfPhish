import tensorflow as tf
import numpy as np
import skimage.io as io
import time
import os
TF_GPU_ALLOCATOR= 'cuda_malloc_async'
os.environ["CUDA_VISIBLE_DEVICES"]=""

def tf_reshape(img):
    """
    output: [1, img.shape[0], img.shape[1], 1] or [img.shape[0], img.shape[1], img.shape[2], 1]
    """
    try:
        img = tf.convert_to_tensor(img)
        return tf.cast(tf.reshape(img, [1, img.shape[0], img.shape[1], 1]), dtype=tf.float32)
    except:
        if type(img) is np.ndarray or type(img) is tf.Tensor:
            if len(img.shape) == 2:
                img = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
            elif len(img.shape) == 3:
                img = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
            elif len(img.shape) == 4:
                img = img
        elif type(img) is list:
            if len(img[0].shape) == 2:
                img = tf.stack([tf_reshape(i) for i in img])
                img = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
            elif len(img[0].shape) == 3:
                img = tf.stack([tf_reshape(i) for i in img])
            elif len(img[0].shape) == 4:
                img = tf.stack(img)
            try:
                img = img.numpy()
                img = tf_reshape(img)
            except:
                raise TypeError("img must be a list, np.ndarray or tf.Tensor")
        img = tf.cast(img, dtype=tf.float32)
        return img

def load_image(path):
    #if .npy, 
    if 'npy' in path:
        img = tf_reshape(np.load(path).astype('float32'))
    elif 'npz' in path:
        img = tf_reshape(np.load(path)['arr_0'].astype('float32'))

    elif 'tif' in path:
        img = tf_reshape(io.imread(path))
    elif 'tiff' in path:
        img = tf_reshape(io.imread(path))
    elif 'png' in path:
        img = tf_reshape(io.imread(path))
    elif 'jpg' in path:
        img = tf_reshape(io.imread(path))
    elif 'collection' in path:
        img = tf_reshape(io.imread_collection(path))
    elif 'ImageCollection' in str(type(path)):
        img = tf_reshape(io.imread_collection(path.files))
    elif 'numpy' in str(type(path)) or 'tensorflow' in str(type(path)):
        img = tf_reshape(path)
    else:
        raise TypeError("img must be a list, np.ndarray or tf.Tensor")
    print(img.shape)
    img = tf.reshape(img, [img.shape[0], 1, img.shape[1], img.shape[2], 1])
    img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img)) * 255
    return img

def write_video(img, file_name = 'data/video/recovered.mp4', fps = None, output_file = None):
    import time
    assert fps is not None
    frame_interval = 1 / fps

    assert len(img.shape) == 5
    px, py = img.shape[2], img.shape[3]
    
    if output_file is None:
        output_file = 'data/video/video.mp4'
    img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img)) * 255
    writer = tf.io.TFRecordWriter(output_file)
    # Write the images to the video file
    for i in range(img.shape[0]):
        # Encode the image as a JPEG
        image_jpeg = tf.image.encode_jpeg(tf.cast(tf.reshape(img[i][0,:,:,0], [px, py, 1]), tf.uint8))

        video_frame = tf.train.SequenceExample(
            context=tf.train.Features(
                feature={
                    'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_jpeg.numpy()])),
                    'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=[b'jpeg'])),
                }
            ),
            feature_lists=tf.train.FeatureLists(
                feature_list={
                    'image/encoded': tf.train.FeatureList(feature=[
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_jpeg.numpy()]))
                    ]),
                }
            )
        )
        writer.write(video_frame.SerializeToString())

        
        # Wait for the specified frame interval
        if i < img.shape[0] - 1:
            time.sleep(frame_interval//5)

    if os.path.exists(file_name):
        randon_number = np.random.randint(1000)
        file_name = file_name.split('.')[0] + str(randon_number) + '.mp4'
    #converting to mp4
    import subprocess  
    subprocess.run(['ffmpeg', '-i', output_file, file_name])

    writer.close()
    return "video written to {}".format(file_name)

def create_video(path, file_name = 'data/video/recovered', fps = None, output_file = None):
    img = load_image(path)
    write_video(img, file_name = file_name, fps = fps, output_file = output_file)
    return "video written to {}".format(file_name)

def __main__():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='data/video/recovered', help='path to the video')
    parser.add_argument('--fps', type=int, default=60, help='fps of the video')
    parser.add_argument('--file_name', type=str, default='data/video/recovered.mp4', help='file name of the video')
    parser.add_argument('--output_file', type=str, default='data/video/video.mp4', help='output file name')
    args = parser.parse_args()
    create_video(args.path, args.file_name, args.fps, output_file = args.output_file)