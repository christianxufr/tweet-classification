import os
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import tensorflow as tf

custom_objects = {
}


class Saving:

    @staticmethod
    def touch_dir(directory: str):
        if not os.path.isdir(directory):
            os.makedirs(directory)
        return directory

    @staticmethod
    def tensorboard_callback(root: str):
        return TensorBoard(log_dir=root,
                           histogram_freq=1)

    @staticmethod
    def model_checkpoint_callback(root: str):
        checkpoint_dir = Saving.touch_dir(os.path.join(root, "checkpoints"))
        return ModelCheckpoint(filepath=os.path.join(checkpoint_dir, "cp{epoch:02d}-{val_loss:.2f}.h5"),
                               save_weights_only=True,
                               verbose=1)

    def __init__(self, directory: str):
        self.root = self.touch_dir(directory)

    def get_latest_run(self, model_name: str):
        def extract_run_number(name: str):
            try:
                return int(name)
            except ValueError:
                return 0
        model_root = self.touch_dir(os.path.join(self.root, model_name))
        latest_id = max(extract_run_number(d) for d in [0, *os.listdir(model_root)])
        return latest_id

    def get_latest_checkpoint(self, model_name: str):
        last_gen = self.get_latest_run(model_name)
        checkpoint = None
        while last_gen > 0 and checkpoint is None:
            path = os.path.join(self.root, model_name, str(last_gen), "checkpoints")
            if os.path.isdir(path):
                checkpoint = os.path.join(path, sorted(os.listdir(path))[-1])
            last_gen -= 1
        return checkpoint

    def get_callbacks(self, model_name: str,
                      tensorboard: bool = True,
                      model_checkpoint: bool = True):
        new_id = self.get_latest_run(model_name) + 1
        generation_root = self.touch_dir(os.path.join(self.root, model_name, str(new_id)))
        callbacks = []
        if tensorboard:
            callbacks.append(self.tensorboard_callback(generation_root))
        if model_checkpoint:
            callbacks.append(self.model_checkpoint_callback(generation_root))
        return callbacks

    def save_model(self, model, model_name: str, override: bool = False):
        self.touch_dir(os.path.join(self.root, model_name))
        path = os.path.join(self.root, model_name, model_name + ".h5")
        if not os.path.isfile(path) or override:
            model.save(path)

    def load_model(self, model_name: str, checkpoint_file: str = None):
        model_path = os.path.join(self.root, model_name, model_name + ".h5")
        if not os.path.isfile(model_path):
            return None
        print("[Saving] Loading model at: " + model_path)
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        if checkpoint_file is None:
            checkpoint_path = self.get_latest_checkpoint(model_name)
        else:
            checkpoint_path = os.path.join(self.root, checkpoint_file)
        if checkpoint_path is not None:
            print("[Saving] Loading weights at: " + checkpoint_path)
            model.load_weights(checkpoint_path)
        return model
