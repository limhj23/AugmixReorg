from tensorflow import keras as K
IMAGE_SIZE = (224,224)

def get_mbv2(num_classes):
    base_model = K.applications.MobileNetV2(
        input_shape=IMAGE_SIZE+(3,),
        alpha=1.0,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        pooling='avg',
    )

    x = base_model.output
    # let's add a fully-connected layer
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = K.layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    return model

def get_resnet50(num_classes):
    base_model = K.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling='avg',
    )
    x = base_model.output
    # let's add a fully-connected layer
    # x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = K.layers.Dense(num_classes, activation='softmax')(x)

    # this is the model we will train
    model = K.models.Model(inputs=base_model.input, outputs=predictions)

    return model

def fine_tune(model, layer_num):
    for layer in model.layers[:layer_num]:
        layer.trainable = False
    for layer in model.layers[layer_num:]:
        layer.trainable = True
    return model

def jsd_loss_fn(y_true, y_pred_clean, y_pred_aug1, y_pred_aug2):
    kld = K.losses.KLDivergence()
    # cross entropy loss that is used for clean images only
    loss = K.losses.CategoricalCrossentropy(y_true, y_pred_clean)

    mixture = (y_pred_clean + y_pred_aug1 + y_pred_aug2) / 3.

    loss += 12. * (kld(y_pred_clean, mixture) + 
                   kld(y_pred_aug1, mixture) +
                   kld(y_pred_aug2, mixture)) / 3.
    return loss