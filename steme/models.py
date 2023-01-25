import keras
import keras.backend as K
import tensorflow as tf

from tensorflow.keras import layers


def spice(sigma, w_tempo, w_recon):
    encoder_filter = 64
    decoder_filter = 32
    kernel_size = 3
    strides = 1
    batch_size = 64
    padding = "same"

    x1 = layers.Input(shape=(128, 1))
    x2 = layers.Input(shape=(128, 1))
    k1 = layers.Input(shape=(1))
    k2 = layers.Input(shape=(1))

    conv1 = layers.Conv1D(
        encoder_filter,
        kernel_size,
        strides,
        padding=padding)
    bn1 = layers.BatchNormalization()
    relu1 = layers.ReLU()
    mp1 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    conv2 = layers.Conv1D(
        encoder_filter * 2,
        kernel_size,
        strides,
        padding=padding)
    bn2 = layers.BatchNormalization()
    relu2 = layers.ReLU()
    mp2 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    conv3 = layers.Conv1D(
        encoder_filter * 4,
        kernel_size,
        strides,
        padding=padding)
    bn3 = layers.BatchNormalization()
    relu3 = layers.ReLU()
    mp3 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    conv4 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    bn4 = layers.BatchNormalization()
    relu4 = layers.ReLU()
    mp4 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    conv5 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    bn5 = layers.BatchNormalization()
    relu5 = layers.ReLU()
    mp5 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    conv6 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    bn6 = layers.BatchNormalization()
    relu6 = layers.ReLU()
    mp6 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    flatten = layers.Flatten()

    phead_dense = layers.Dense(48)
    phead_y = layers.Dense(1, activation="sigmoid")

    dense48 = layers.Dense(48)
    reshape = layers.Reshape((48, 1))

    dtconv1 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    dbn1 = layers.BatchNormalization()
    drelu1 = layers.ReLU()
    dmp1 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    dtconv2 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    dbn2 = layers.BatchNormalization()
    drelu2 = layers.ReLU()
    dmp2 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    dtconv3 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding)
    dbn3 = layers.BatchNormalization()
    drelu3 = layers.ReLU()
    dmp3 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    dtconv4 = layers.Conv1DTranspose(
        decoder_filter * 4,
        kernel_size,
        strides,
        padding=padding)
    dbn4 = layers.BatchNormalization()
    drelu4 = layers.ReLU()
    dmp4 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)
    dtconv5 = layers.Conv1DTranspose(
        decoder_filter * 2,
        kernel_size,
        strides,
        padding=padding)
    dbn5 = layers.BatchNormalization()
    drelu5 = layers.ReLU()
    dmp5 = layers.MaxPool1D(pool_size=3, strides=2, padding=padding)

    dreshape = layers.Reshape((128, 1))

    embd1 = conv1(x1)
    embd1 = relu1(embd1)
    embd1 = bn1(embd1)
    embd1 = mp1(embd1)
    embd1 = conv2(embd1)
    embd1 = relu2(embd1)
    embd1 = bn2(embd1)
    embd1 = mp2(embd1)
    embd1 = conv3(embd1)
    embd1 = relu3(embd1)
    embd1 = bn3(embd1)
    embd1 = mp3(embd1)
    embd1 = conv4(embd1)
    embd1 = relu4(embd1)
    embd1 = bn4(embd1)
    embd1 = mp4(embd1)
    embd1 = conv5(embd1)
    embd1 = relu5(embd1)
    embd1 = bn5(embd1)
    embd1 = mp5(embd1)
    embd1 = conv6(embd1)
    embd1 = relu6(embd1)
    embd1 = bn6(embd1)
    embd1 = mp6(embd1)
    embd1 = flatten(embd1)

    embd2 = conv1(x2)
    embd2 = relu1(embd2)
    embd2 = bn1(embd2)
    embd2 = mp1(embd2)
    embd2 = conv2(embd2)
    embd2 = relu2(embd2)
    embd2 = bn2(embd2)
    embd2 = mp2(embd2)
    embd2 = conv3(embd2)
    embd2 = relu3(embd2)
    embd2 = bn3(embd2)
    embd2 = mp3(embd2)
    embd2 = conv4(embd2)
    embd2 = relu4(embd2)
    embd2 = bn4(embd2)
    embd2 = mp4(embd2)
    embd2 = conv5(embd2)
    embd2 = relu5(embd2)
    embd2 = bn5(embd2)
    embd2 = mp5(embd2)
    embd2 = conv6(embd2)
    embd2 = relu6(embd2)
    embd2 = bn6(embd2)
    embd2 = mp6(embd2)
    embd2 = flatten(embd2)

    y1 = phead_dense(embd1)
    y1 = phead_y(y1)
    y2 = phead_dense(embd2)
    y2 = phead_y(y2)

    xhat1 = dense48(y1)
    xhat1 = reshape(xhat1)
    xhat1 = dtconv1(xhat1)
    xhat1 = drelu1(xhat1)
    xhat1 = dbn1(xhat1)
    xhat1 = dmp1(xhat1)
    xhat1 = dtconv2(xhat1)
    xhat1 = drelu2(xhat1)
    xhat1 = dbn2(xhat1)
    xhat1 = dmp2(xhat1)
    xhat1 = dtconv3(xhat1)
    xhat1 = drelu3(xhat1)
    xhat1 = dbn3(xhat1)
    xhat1 = dmp3(xhat1)
    xhat1 = dtconv4(xhat1)
    xhat1 = drelu4(xhat1)
    xhat1 = dbn4(xhat1)
    xhat1 = dmp4(xhat1)
    xhat1 = dtconv5(xhat1)
    xhat1 = drelu5(xhat1)
    xhat1 = dbn5(xhat1)
    xhat1 = dmp5(xhat1)
    xhat1 = dreshape(xhat1)

    xhat2 = dense48(y2)
    xhat2 = reshape(xhat2)
    xhat2 = dtconv1(xhat2)
    xhat2 = drelu1(xhat2)
    xhat2 = dbn1(xhat2)
    xhat2 = dmp1(xhat2)
    xhat2 = dtconv2(xhat2)
    xhat2 = drelu2(xhat2)
    xhat2 = dbn2(xhat2)
    xhat2 = dmp2(xhat2)
    xhat2 = dtconv3(xhat2)
    xhat2 = drelu3(xhat2)
    xhat2 = dbn3(xhat2)
    xhat2 = dmp3(xhat2)
    xhat2 = dtconv4(xhat2)
    xhat2 = drelu4(xhat2)
    xhat2 = dbn4(xhat2)
    xhat2 = dmp4(xhat2)
    xhat2 = dtconv5(xhat2)
    xhat2 = drelu5(xhat2)
    xhat2 = dbn5(xhat2)
    xhat2 = dmp5(xhat2)
    xhat2 = dreshape(xhat2)

    model = tf.keras.Model([x1, x2, k1, k2], [xhat1, xhat2, y1, y2])

    h = tf.keras.losses.Huber(
        delta=0.25 * sigma,
        reduction="sum_over_batch_size")

    e_t = K.abs((y1 - y2) - sigma * (k2 - k1))

    loss_tempo = h(e_t, 0) * w_tempo
    # https://math.stackexchange.com/questions/2690199/should-the-2-in-l-2-norm-notation-be-a-subscript-or-superscript
    loss_recon = K.mean(K.mean(K.square(x1 - xhat1) +
                        K.square(x2 - xhat2), axis=1)) * w_recon

    model.add_loss(loss_tempo)
    model.add_loss(loss_recon)

    model.add_metric(loss_tempo, name="tempo_loss")
    model.add_metric(loss_recon, name="reconstruction_loss")

    return model


def convolutional_autoencoder(sigma, w_tempo, w_recon):
    encoder_filter = 64
    decoder_filter = 64
    kernel_size = 3
    strides = 2
    padding = "same"
    activation = "relu"
    batch_size = 64

    x1 = layers.Input(shape=(128, 1))
    x2 = layers.Input(shape=(128, 1))
    k1 = layers.Input(shape=(1))
    k2 = layers.Input(shape=(1))

    # encoder
    conv1 = layers.Conv1D(
        encoder_filter,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    conv2 = layers.Conv1D(
        encoder_filter * 2,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    conv3 = layers.Conv1D(
        encoder_filter * 4,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    conv4 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    conv5 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    conv6 = layers.Conv1D(
        encoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    flatten = layers.Flatten()

    # tempo head
    phead_dense = layers.Dense(48)
    phead_y = layers.Dense(1, activation="sigmoid")

    # decoder
    reshape = layers.Reshape((1, 1))

    dtconv1 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dtconv2 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dtconv3 = layers.Conv1DTranspose(
        decoder_filter * 8,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dtconv4 = layers.Conv1DTranspose(
        decoder_filter * 4,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dtconv5 = layers.Conv1DTranspose(
        decoder_filter * 2,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dtconv6 = layers.Conv1DTranspose(
        decoder_filter,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)
    dreshape = layers.Conv1DTranspose(
        1,
        kernel_size,
        strides,
        padding=padding,
        activation=activation)

    # encoder 1
    embd1 = conv1(x1)
    embd1 = conv2(embd1)
    embd1 = conv3(embd1)
    embd1 = conv4(embd1)
    embd1 = conv5(embd1)
    embd1 = conv6(embd1)
    embd1 = flatten(embd1)

    y1 = phead_dense(embd1)
    y1 = phead_y(y1)

    xhat1 = reshape(y1)
    xhat1 = dtconv1(xhat1)
    xhat1 = dtconv2(xhat1)
    xhat1 = dtconv3(xhat1)
    xhat1 = dtconv4(xhat1)
    xhat1 = dtconv5(xhat1)
    xhat1 = dtconv6(xhat1)
    xhat1 = dreshape(xhat1)

    embd2 = conv1(x2)
    embd2 = conv2(embd2)
    embd2 = conv3(embd2)
    embd2 = conv4(embd2)
    embd2 = conv5(embd2)
    embd2 = conv6(embd2)
    embd2 = flatten(embd2)

    y2 = phead_dense(embd2)
    y2 = phead_y(y2)

    xhat2 = reshape(y2)
    xhat2 = dtconv1(xhat2)
    xhat2 = dtconv2(xhat2)
    xhat2 = dtconv3(xhat2)
    xhat2 = dtconv4(xhat2)
    xhat2 = dtconv5(xhat2)
    xhat2 = dtconv6(xhat2)
    xhat2 = dreshape(xhat2)

    model = tf.keras.Model([x1, x2, k1, k2], [xhat1, xhat2, y1, y2])

    h = tf.keras.losses.Huber(
        delta=0.25 * sigma,
        reduction="sum_over_batch_size")

    e_t = K.abs((y1 - y2) - sigma * (k2 - k1))

    loss_tempo = h(e_t, 0) * w_tempo
    # https://math.stackexchange.com/questions/2690199/should-the-2-in-l-2-norm-notation-be-a-subscript-or-superscript
    loss_recon = K.mean(K.mean(K.square(x1 - xhat1) +
                        K.square(x2 - xhat2), axis=1)) * w_recon

    model.add_loss(loss_tempo)
    model.add_loss(loss_recon)

    model.add_metric(loss_tempo, name="tempo_loss")
    model.add_metric(loss_recon, name="reconstruction_loss")

    return model


def overfit_to_one_sample(model):
    print("Overfitting to one sample")
    s1, sh1, s2, sh2, _ = audio.overfit_sample()
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE))
    model.fit([s1, s1, sh1, sh1], [s1, s1, sh1, sh1], epochs=50)
    recon1, outshift1, recon2, outshift2 = model.predict([s1, s1, sh1, sh1])

    assert np.allclose(recon1, s1, atol=0.1), "Model does not overfit"
