"""
Improved Model Architecture for AIROGS glaucoma detection
Fixes over-regularization issues from previous training
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50, EfficientNetB3


def create_improved_model(
    backbone="efficientnet-b0", input_shape=(384, 384, 3), num_classes=1
):
    """
    Create IMPROVED model for glaucoma detection

    Key changes from baseline:
    - REDUCED dropout to prevent over-regularization
    - LIGHTER regularization (L2 reduced)
    - Better initialization strategy
    - Properly configured for fine-tuning

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

    # Start with frozen backbone - will unfreeze later
    base_model.trainable = False

    # Build model
    inputs = layers.Input(shape=input_shape)

    # Backbone
    x = base_model(inputs, training=False)

    # IMPROVED Classification head with REDUCED regularization
    # Previous model had 0.5, 0.5, 0.4 dropout - TOO MUCH!
    x = layers.Dropout(0.3)(x)  # Reduced from 0.5
    x = layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0005),  # Reduced from 0.001
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)  # Added for stability

    x = layers.Dropout(0.3)(x)  # Reduced from 0.5
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(0.0005),  # Reduced from 0.001
        kernel_initializer="he_normal",
    )(x)
    x = layers.BatchNormalization()(x)  # Added for stability

    x = layers.Dropout(0.2)(x)  # Reduced from 0.4

    # Output layer (sigmoid for binary classification)
    outputs = layers.Dense(
        num_classes,
        activation="sigmoid",
        name="glaucoma_output",
        kernel_initializer="glorot_uniform",
    )(x)

    model = models.Model(
        inputs=inputs, outputs=outputs, name=f"AIROGS_improved_{backbone}"
    )

    return model


def unfreeze_backbone(model, num_layers=50):
    """
    Unfreeze the last num_layers of the backbone for fine-tuning

    Args:
        model: Keras model
        num_layers: Number of layers to unfreeze from the end
    """
    # Find the base model (usually the first layer that's a Model)
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        print("⚠️  No base model found to unfreeze")
        return

    # Freeze all layers first
    base_model.trainable = True

    # Freeze early layers, unfreeze later ones
    total_layers = len(base_model.layers)
    freeze_until = max(0, total_layers - num_layers)

    for i, layer in enumerate(base_model.layers):
        if i < freeze_until:
            layer.trainable = False
        else:
            layer.trainable = True

    print(f"✅ Unfroze last {num_layers} layers of backbone")
    print(
        f"   Total layers: {total_layers}, Frozen: {freeze_until}, Trainable: {num_layers}"
    )


def get_callbacks(model_name, config):
    """
    Create training callbacks with improved settings

    Args:
        model_name: Name for saving checkpoints
        config: Configuration module

    Returns:
        List of Keras callbacks
    """
    import os
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        ReduceLROnPlateau,
        CSVLogger,
        TensorBoard,
        LearningRateScheduler,
    )

    callbacks = []

    # Model checkpoint - save best model
    checkpoint_path = os.path.join(config.MODELS_DIR, f"{model_name}_best.keras")
    checkpoint = ModelCheckpoint(
        checkpoint_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1
    )
    callbacks.append(checkpoint)

    # Early stopping
    early_stop = EarlyStopping(
        monitor="val_auc",
        mode="max",
        patience=config.EARLY_STOPPING_PATIENCE,
        verbose=1,
        restore_best_weights=True,
    )
    callbacks.append(early_stop)

    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(
        monitor="val_auc",
        mode="max",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1,
    )
    callbacks.append(reduce_lr)

    # CSV logger
    csv_path = os.path.join(config.LOGS_DIR, f"{model_name}_training.csv")
    csv_logger = CSVLogger(csv_path)
    callbacks.append(csv_logger)

    # TensorBoard
    tb_path = os.path.join(config.LOGS_DIR, model_name)
    tensorboard = TensorBoard(log_dir=tb_path, histogram_freq=0)
    callbacks.append(tensorboard)

    # Learning rate warmup (if enabled)
    if hasattr(config, "USE_LR_WARMUP") and config.USE_LR_WARMUP:

        def lr_schedule(epoch, lr):
            if epoch < config.WARMUP_EPOCHS:
                # Linear warmup
                warmup_lr = config.LEARNING_RATE * (epoch + 1) / config.WARMUP_EPOCHS
                print(f"  Warmup epoch {epoch+1}: LR = {warmup_lr:.2e}")
                return warmup_lr
            return lr

        warmup_callback = LearningRateScheduler(lr_schedule)
        callbacks.append(warmup_callback)

    return callbacks
