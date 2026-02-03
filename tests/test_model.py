import pytest
#import warnings
from tqdm import tqdm
from loguru import logger
import tensorflow as tf
import iraklis7_nn.dataset as dataset
import iraklis7_nn.features as features


def test_model():
    iterations = 1
    loss_threshold = 0.10
    accuracy_threshold = 0.95

    for i in tqdm(range(iterations)):
        logger.info(f"Starting iteration {i}")

        logger.info("Retrieving datasets with shuffle enabled...")
        (ds_train, ds_val, ds_test), ds_info = dataset.make_dataset_random()
        
        ds_train_size = len(ds_train)
        ds_val_size = len(ds_val)
        ds_test_size = len(ds_test)
        logger.info(
            f"Train size: {ds_train_size} Validation size: {ds_val_size} Test size; {ds_test_size}"
        )
        # Check the size of the datasets
        assert ds_train_size + ds_val_size + ds_test_size == 70000

        logger.info("Optimizing datasets...")
        # Optimize datasets
        ds_train = features.optimise_ds(ds_train, ds_info, is_train=True)
        ds_val = features.optimise_ds(ds_val, ds_info, is_train=False)
        ds_test = features.optimise_ds(ds_test, ds_info, is_train=False)

        # Create the model
        model = tf.keras.models.Sequential(
            [
                # Need to get rid of warning by using Flatten and input_shape
                tf.keras.layers.Flatten(input_shape=(28, 28)),
                tf.keras.layers.Dense(
                    128,
                    activation="relu",
                ),
                tf.keras.layers.Dense(10),
            ]
        )

        # Print model summary
        model.summary()

        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        # Fit the model
        model.fit(
            ds_train,
            epochs=12,
            validation_data=ds_val,
        )

        # Make predictions on test dataset
        y_pred_full = model.predict(ds_test)

        # Report on metrics
        result = model.get_metrics_result()
        logger.info(f"Loss: {result['loss']} Accuracy: {result['sparse_categorical_accuracy']}")
        assert result['loss'] < loss_threshold, f"Loss exceeded threshold of {loss_threshold}"
        assert result['sparse_categorical_accuracy'] > accuracy_threshold, f"Accuracy is under the threshold of {accuracy_threshold}"
        
        logger.success(f"Iteration {i} was succesful")