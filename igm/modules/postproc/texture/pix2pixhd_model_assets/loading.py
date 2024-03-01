import tensorflow as tf
import os

def load_model_test(checkpoint, checkpoint_dir):
    
    model_path = tf.train.latest_checkpoint(checkpoint_dir)

    try:
        # print(model_path, checkpoint_dir)
        checkpoint.restore(model_path).assert_consumed() #.expect_partial()
        print("Successfully restored model.")
    except AssertionError as error:
        print(error)
    except:
        raise ImportError("Could not find model to load. Check pretrain model path.")
