import os

import tensorflow as tf


class ModelSession(object):
    """
    A session of a TensorFlow model that may be serialized.

    The model's graph structure is defined by overriding the create_graph function.
    """

    def __init__(self, session, saver, args):
        """
        Create a model session.

        Do not call this constructor directly. To instantiate a ModelSession object, use the create and restore class
        methods.

        :param session: the session in which this model is running
        :type session: tf.Session
        :param saver: object used to serialize this session
        :type saver: tf.Saver
        """
        self.session, self.saver, self.args = session, saver, args

    @classmethod
    def create(cls, **kwargs):
        """
        Create a new model session.

        :param kwargs: optional graph parameters
        :type kwargs: dict
        :return: new model session
        :rtype: ModelSession
        """
        session = tf.Session()
        with session.graph.as_default():
            cls.create_graph(**kwargs)
        session.run(tf.initialize_all_variables())
        return cls(session, tf.train.Saver())

    @staticmethod
    def create_graph(**kwargs):
        """
        Override this function to define a TensorFlow graph.

        For example, the following creates a graph containing a single variable named "x".

            def create_graph(init = 1.0):
                tf.Variable(initial_value = init, name="x")

        :param kwargs: optional graph parameters
        :type kwargs: dict
        """
        raise NotImplementedError()

    @classmethod
    def restore(cls, checkpoint_directory):
        """
        Restore a serialized model session.

        :param checkpoint_directory:  directory containing checkpoint files
        :type checkpoint_directory: str
        :return: session restored from the latest checkpoint file
        :rtype: ModelSession
        """
        session = tf.Session()
        # Deserialize the graph.
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_directory)
        if checkpoint_file is None:
            raise ValueError("Invalid checkpoint directory %s" % checkpoint_directory)
        saver = tf.train.import_meta_graph(checkpoint_file + ".meta")
        saver.restore(session, checkpoint_file)
        # Subsequent saves of this model during this session must be done with the same saver object that was used to
        # deserialize it.
        return cls(session, saver)

    def save(self, checkpoint_directory):
        """
        Save the current model session to a checkpoint file.

        If the graph defines an "iteration" variable its value will be used for the global step in the checkpoint name.

        :param checkpoint_directory:  directory containing checkpoint files
        :type checkpoint_directory: str
        :return: path to the new checkpoint file
        :rtype: str
        """
        try:
            iteration = self.session.graph.get_tensor_by_name("iteration:0")
            global_step = self.session.run(iteration)
        except KeyError:
            global_step = None
        path = self.saver.save(self.session, os.path.join(checkpoint_directory, "model.ckpt"), global_step=global_step)
        return path
