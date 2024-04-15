import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="test.py")
interpreter.allocate_tensor()

tf.saved_model.save(interpreter,"tf_model")
