### DO NOT CHANGE THIS FILE ###
# This file is responsbile for mapping the predictions in and out of the inference server
# changes here will likely break or invalidate your submission.

import json
import shutil
import os
from os.path import exists
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
import pandas as pd
from PIL import Image
import io

IMG_SIZE = 224
LBL = dict(zip(['B_BSMUT1', 'B_CLEV5B', 'B_DISTO', 'B_GRMEND', 'B_HDBARL',
       'B_PICKLD', 'B_SKINED', 'B_SOUND', 'B_SPRTED', 'B_SPTMLD',
       'O_GROAT', 'O_HDOATS', 'O_SEPAFF', 'O_SOUND', 'O_SPOTMA',
       'WD_RADPODS', 'WD_RYEGRASS', 'WD_SPEARGRASS', 'WD_WILDOATS',
       'W_DISTO', 'W_FLDFUN', 'W_INSDA2', 'W_PICKLE', 'W_SEVERE',
       'W_SOUND', 'W_SPROUT', 'W_STAIND', 'W_WHITEG'], range(28)))
cls_map = dict(zip(LBL.values(),LBL.keys()))

def init_dir(pth):
    if exists(pth):
        shutil.rmtree(pth)
        os.makedirs(pth)
    else:
        os.makedirs(pth)

def input_handler(data, context):
    """ Pre-process request input before it is sent to TensorFlow Serving REST API

    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details

    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """
    if context.request_content_type == 'application/x-image':
        payload = data.read()

        img = Image.open(io.BytesIO(payload))
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        img_array = image.img_to_array(img)
        img_array = img_array.astype(np.uint8)
        
        img_preprocessed = preprocess_input(img_array)[None, :]

        return json.dumps({"instances": np.array(img_preprocessed).tolist()})
    else:
        _return_error(415, 'Unsupported content type was "{}"'.format(
            context.request_content_type or 'Unknown'))

def output_handler(response, context):
    """Post-process TensorFlow Serving output before it is returned to the client.

    Args:
        response (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details

    Returns:
        (bytes, string): data to return to client, response content type
    """
    print("Output handler")
    
    if response.status_code != 200:
        _return_error(response.status_code, response.content.decode('utf-8'))
    response_content_type = context.accept_header
    content = response.content

    predictions = json.loads(content.decode('UTF-8'))
    predictions = np.array(predictions["predictions"])
    res = []
    for pred in predictions:
        top3 = (-pred).argsort()[:3]
        res.append({'file_name': 'no-filename', 'path': 'no-path', 'cls': 'actual', 'prediction':top3[0],  'proba_1':pred[top3[0]], 'prediction2':top3[1], 'proba_2':pred[top3[1]],  'prediction3':top3[2], 'proba_3':pred[top3[2]]})

    image_index = pd.DataFrame(res)
    image_index['prediction'] = image_index.prediction.map(cls_map)
    image_index['prediction2'] = image_index.prediction2.map(cls_map)
    image_index['prediction3'] = image_index.prediction3.map(cls_map)
    return image_index.to_csv(index=False, header=False), response_content_type

def _return_error(code, message):
    raise ValueError('Error: {}, {}'.format(str(code), message))
