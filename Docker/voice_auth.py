import numpy as np
from scipy.spatial.distance import euclidean, cosine 
import warnings
from keras.models import load_model
import logging
logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings("ignore")
import os
import tempfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)
#IMPORT USER-DEFINED FUNCTIONS
from feature_extraction import get_embedding, get_embeddings_from_list_file
from preprocess import get_fft_spectrum
import parameters as p


def enroll(name,file):
    """Enroll a user with an audio file
        inputs: str (Name of the person to be enrolled and registered)
                str (Path to the audio file of the person to enroll)
        outputs: None"""

    print("Loading model weights from [{}]....".format(p.MODEL_FILE))
    try:
        model = load_model(p.MODEL_FILE)
    except:
        print("Failed to load weights from the weights file, please ensure *.pb file is present in the MODEL_FILE directory")
        exit()
    
    try:
        print("Processing enroll sample....")
        enroll_result = get_embedding(model, file, p.MAX_SEC)
        enroll_embs = np.array(enroll_result.tolist())
        speaker = name
    except:
        print("Error processing the input audio file. Make sure the path.")
    try:
        np.save(os.path.join(p.EMBED_LIST_FILE,speaker +".npy"), enroll_embs)
        return True
    except:
        return False


def recognize(file):
    """Recognize the input audio file by comparing to saved users' voice prints
        inputs: file (Audio file of unknown person to recognize)
        outputs: str (Name of the person recognized)"""
    
    # Save the file to a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    file.save(temp_file.name)

    if os.path.exists(p.EMBED_LIST_FILE):
        embeds = os.listdir(p.EMBED_LIST_FILE)
    if len(embeds) is 0:
        return False
    try:
        model = load_model(p.MODEL_FILE)

    except:
        return False
        
    distances = {}
    test_result = get_embedding(model, temp_file.name, p.MAX_SEC)
    print(test_result)
    test_embs = np.array(test_result.tolist())
    for emb in embeds:
        enroll_embs = np.load(os.path.join(p.EMBED_LIST_FILE,emb))
        speaker = emb.replace(".npy","")
        distance = euclidean(test_embs, enroll_embs)
        distances.update({speaker:distance})
    max_distance = max(list(distances.values()))
    print(f"Max distance: {max_distance}, Threshold: {p.THRESHOLD}")
    if max_distance < p.THRESHOLD:
        return False
    else:
        return True


