#!/usr/bin/env python3
"""
    Evaluates the model on the test set
"""

import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
        Evaluates the model on the test set

        Args:
            X: input data
            Y: input labels
            save_path: path to the model

        Returns:
            y_pred: predicted labels
            accuracy: accuracy of the prediction
            loss: loss of the prediction
    """

    with tf.Session() as session:
        store = tf.train.import_meta_graph("{}.meta".format(save_path))
        store.restore(session, save_path)
        graph = tf.get_default_graph()

        x = graph.get_collection('x')[0]
        y = graph.get_collection('y')[0]
        y_pred = graph.get_collection('y_pred')[0]
        loss = graph.get_collection('loss')[0]
        accuracy = graph.get_collection('accuracy')[0]

        return session.run(
            [
                y_pred,
                accuracy,
                loss
            ],
            feed_dict={x: X, y: Y}
        )
