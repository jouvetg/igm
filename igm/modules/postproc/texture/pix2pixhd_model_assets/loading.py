import tensorflow as tf
import os

def load_model_test(checkpoint, checkpoint_dir):
    
    # if model.opt.load_pretrain_dir == '':
    #     print("Training from scratch.")
    #     return 0
    # else:
    #     if model.opt.which_epoch == 'latest':
    model_path = tf.train.latest_checkpoint(checkpoint_dir)
    #     else:
    # model_path = os.path.join(model_path)

    try:
        checkpoint.restore(model_path).expect_partial()
        print("Successfully restored model.")
    except AssertionError as error:
        print(error)
    except:
        raise ImportError("Could not find model to load. Check pretrain model path.")
