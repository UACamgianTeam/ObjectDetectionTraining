import tensorflow as tf
import numpy as np

from typing import List, Tuple


def run_inference(test_images_np: List, test_images_dict: dict, detection_model: any) -> Tuple[dict, List, List]:
    """ Runs the detection process on each of the windows in the test set and
    stores the results in test_windows_dict
    """
    predicted_boxes = [None] * len(test_images_np)
    predicted_scores = [None] * len(test_images_np)
    for image_id, image_info in test_images_dict.items():
        for window_id, window_info in image_info['windows'].items():
            window = test_images_np[window_id]
            input_tensor = tf.convert_to_tensor(np.expand_dims(window, axis=0), dtype=tf.float32)
            detections = detect(input_tensor, detection_model)
            predicted_boxes[window_id] = detections['detection_boxes'][0].numpy()
            predicted_scores[window_id] = detections['detection_scores'][0].numpy()
            test_images_dict[image_id]['windows'][window_id]['predicted_boxes'] = detections['detection_boxes'][
                0].numpy()
            test_images_dict[image_id]['windows'][window_id]['predicted_scores'] = detections['detection_scores'][
                0].numpy()

    print(test_images_dict)
    return (test_images_dict, predicted_boxes, predicted_scores)


# Again, uncomment this decorator if you want to run inference eagerly
@tf.function
def detect(input_tensor: any, detection_model: any) -> dict:
    """Run detection on an input image.
    Args:
      input_tensor: A [1, height, width, 3] Tensor of type tf.float32.
        Note that height and width can be anything since the image will be
        immediately resized according to the needs of the model within this
        function.
      detection_model: An object detection model used for inference.
    Returns:
      A dict containing 3 Tensors (`detection_boxes`, `detection_classes`,
        and `detection_scores`).
    """
    preprocessed_image, shapes = detection_model.preprocess(input_tensor)
    prediction_dict = detection_model.predict(preprocessed_image, shapes)
    return detection_model.postprocess(prediction_dict, shapes)
