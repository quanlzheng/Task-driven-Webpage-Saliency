def readtestdata(file_name_list):
    # output file name string to a queue
    filename_queue = tf.train.string_input_producer(file_name_list, capacity=10, shuffle=False)
    # create a reader from file queue
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # get feature from serialized example
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           "raw_image": tf.FixedLenFeature([], tf.string, ),
                                       })

    image_shape = [1080, 1920, 3]

    image = tf.reshape(tf.decode_raw(features['raw_image'], tf.uint8), image_shape)

    image_resize = tf.image.resize_images(image, [224, 224])


    batch_img = tf.train.batch([image_resize], 1,capacity=64)
    return batch_img