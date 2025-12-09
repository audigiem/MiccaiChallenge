"""
Model architecture for AIROGS glaucoma detection
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, EfficientNetB3
import config
import os

def create_baseline_model(
    backbone="efficientnet-b0", input_shape=(384, 384, 3), num_classes=1
):
    """
    Create baseline model for glaucoma detection

    Args:
        backbone: Model backbone ('efficientnet-b0', 'resnet50', 'efficientnet-b3')
        input_shape: Input image shape
        num_classes: Number of output classes (1 for binary classification)

    Returns:
        Compiled Keras model
    """

    # Load pretrained backbone
    if backbone == "efficientnet-b0":
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone == "efficientnet-b3":
        base_model = EfficientNetB3(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone == "resnet50":
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Unfreeze last layers for fine-tuning
    # Initially freeze all layers
    base_model.trainable = True

    # Build model
    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = base_model(inputs, training=False)

    # Classification head with stronger regularization
    x = layers.Dropout(0.5)(x)  # Increased from 0.3
    x = layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.5)(x)  # Increased from 0.3
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.Dropout(0.4)(x)  # Increased from 0.2

    # Output layer (sigmoid for binary classification)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="glaucoma_output")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"AIROGS_{backbone}")

    return model


def create_model_with_gradability(
    backbone="efficientnet-b0", input_shape=(384, 384, 3)
):
    """
    Create model with dual outputs: glaucoma detection + gradability assessment
    (For future improvement - Week 2)

    Args:
        backbone: Model backbone
        input_shape: Input image shape

    Returns:
        Compiled Keras model with two outputs
    """

    # Load pretrained backbone
    if backbone == "efficientnet-b0":
        base_model = EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    elif backbone == "resnet50":
        base_model = ResNet50(
            include_top=False,
            weights="imagenet",
            input_shape=input_shape,
            pooling="avg",
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    base_model.trainable = True

    # Build model
    inputs = layers.Input(shape=input_shape)

    # Shared backbone
    features = base_model(inputs, training=False)

    # Glaucoma detection head
    x1 = layers.Dropout(0.3)(features)
    x1 = layers.Dense(256, activation="relu")(x1)
    x1 = layers.Dropout(0.2)(x1)
    glaucoma_output = layers.Dense(1, activation="sigmoid", name="glaucoma")(x1)

    # Gradability assessment head (for robustness)
    x2 = layers.Dropout(0.3)(features)
    x2 = layers.Dense(128, activation="relu")(x2)
    x2 = layers.Dropout(0.2)(x2)
    gradability_output = layers.Dense(1, activation="sigmoid", name="gradability")(x2)

    model = models.Model(
        inputs=inputs,
        outputs=[glaucoma_output, gradability_output],
        name=f"AIROGS_{backbone}_dual",
    )

    return model


def compile_model(model, learning_rate=1e-4, class_weights=None):
    """
    Compile model with optimizer, loss, and metrics

    Args:
        model: Keras model
        learning_rate: Learning rate
        class_weights: Dictionary of class weights for imbalanced data
    """

    # Optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Loss function with class weights
    if class_weights is not None:
        # Custom weighted binary crossentropy
        def weighted_binary_crossentropy(y_true, y_pred):
            # Get weights based on true labels
            weights = tf.where(tf.equal(y_true, 1), class_weights[1], class_weights[0])
            # Calculate binary crossentropy
            bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            # Apply weights
            weighted_bce = tf.multiply(bce, weights)
            return tf.reduce_mean(weighted_bce)

        loss = weighted_binary_crossentropy
    else:
        loss = "binary_crossentropy"

    # Metrics
    metrics = [
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model


def get_callbacks(model_name="model", patience=5):
    """
    Get training callbacks

    Args:
        model_name: Name for saved model
        patience: Patience for early stopping

    Returns:
        List of callbacks
    """
    # Create directories
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.LOGS_DIR, exist_ok=True)

    callbacks = [
        # Model checkpoint - save best model based on val_auc
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.MODELS_DIR, f"{model_name}_best.keras"),
            monitor="val_auc",
            mode="max",  # Maximize AUC
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        # Early stopping based on val_auc
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",  # Maximize AUC
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",  # Maximize AUC
            factor=config.REDUCE_LR_FACTOR,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, model_name),
            histogram_freq=0,
            write_graph=True,
            update_freq="epoch",
        ),
        # CSV logger
        tf.keras.callbacks.CSVLogger(
            os.path.join(config.LOGS_DIR, f"{model_name}_training.csv"),
            separator=",",
            append=False,
        ),
    ]

    return callbacks
