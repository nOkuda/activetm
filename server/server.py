"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import json
import numpy
import numbers
import tempfile
import os
import random
import uuid

import ankura

from activetm.active import select
from activetm.tech.sampler import slda

app = flask.Flask(__name__, static_url_path='')

FILENAME = '/local/cojoco/git/amazon/amazon.txt'
ENGL_STOP = '/local/cojoco/git/jeffData/stopwords/english.txt'
LABELS = '/local/cojoco/git/amazon/amazon.response'
PIPELINE = [(ankura.read_file, FILENAME, ankura.tokenize.simple),
            (ankura.filter_stopwords, ENGL_STOP),
            (ankura.filter_rarewords, 100),
            (ankura.filter_commonwords, 2000)] 

CAND_SIZE = 500
SEED = 531

NUM_TOPICS = 20
ALPHA = 0.1
BETA = 0.01
VAR = 0.1
NUM_TRAIN = 5
NUM_SAMPLES_TRAIN = 5
TRAIN_BURN = 50
TRAIN_LAG = 50
NUM_SAMPLES_PREDICT = 5
PREDICT_BURN = 10
PREDICT_LAG = 5

START_LABELED = 50
END_LABELED = 100
LABEL_INCREMENT = 10

TEST_SIZE = 200

rng = random.Random(SEED)

@ankura.util.memoize
@ankura.util.pickle_cache('amazon.pickle')
def get_dataset():
    """Gets the dataset object using the Ankura code base"""
    dataset = ankura.run_pipeline(PIPELINE)
    return dataset


@app.route('/random_doc')
def get_random_doc():
    """Gets a document using the random method"""
    # Setup
    dataset = get_dataset()
    pre_labels = {}
    with open(LABELS, 'r') as ifh:
        for line in ifh:
            data = line.strip().split()
            pre_labels[data[0]] = float(data[1])
    labels = []
    for doc_id in range(dataset.num_docs):
        labels.append(pre_labels[dataset.titles[doc_id]])

    # Initialize sets
    shuffled_doc_ids = list(range(dataset.num_docs))
    rng.shuffle(shuffled_doc_ids)
    test_doc_ids = shuffled_doc_ids[:TEST_SIZE]
    test_labels = []
    test_words = []
    for t in test_doc_ids:
        test_labels.append(labels[t])
        test_words.append(dataset.doc_tokens(t))
    test_labels_mean = numpy.mean(test_labels)
    labeled_doc_ids = shuffled_doc_ids[TEST_SIZE:TEST_SIZE+START_LABELED]
    known_labels = []
    for t in labeled_doc_ids:
        known_labels.append(labels[t])
    unlabeled_doc_ids = set(shuffled_doc_ids[TEST_SIZE+START_LABELED:])

    # Create model
    model = slda.SamplingSLDA(rng, NUM_TOPICS, ALPHA, BETA, VAR,
            NUM_TRAIN, NUM_SAMPLES_TRAIN, TRAIN_BURN, TRAIN_LAG,
            NUM_SAMPLES_PREDICT, PREDICT_BURN, PREDICT_LAG)

    # Return document number
    selection = select.random(dataset, labeled_doc_ids, 
                            list(unlabeled_doc_ids), model, rng, 1)[0]
    document = None
    with open(FILENAME, 'r') as documents:
        for i, doc in enumerate(documents):
            if i == selection-1:
                document = doc.split('\t')[1].strip()
                break
    return flask.jsonify(document=document,docnumber=selection)


@app.route('/')
def serve_landing_page():
    """Serves the Active Topic Modeling UI"""
    return flask.send_from_directory('static','index.html')

@app.route('/docs.html')
def serve_ui():
    return flask.send_from_directory('static','docs.html')


@app.route('/scripts/script.js')
def serve_ui_js():
    """Serves the Javascript for the Active TM UI"""
    return flask.send_from_directory('static/scripts','script.js')


@app.route('/scripts/js.cookie.js')
def serve_cookie_script():
    """Serves the Javascript cookie script"""
    return flask.send_from_directory('static/scripts','js.cookie.js')


@app.route('/stylesheets/style.css')
def serve_ui_css():
    """Serves the CSS file for the Active TM UI"""
    return flask.send_from_directory('static/stylesheets','style.css')


@app.route('/uuid')
def get_uid():
    """Sends a UUID to the client"""
    data = {"id": uuid.uuid4()}
    return flask.jsonify(data)

@app.route('/rated', methods=['POST'])
def get_rating():
    """Receives and saves a user rating to a specific user's file"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    with open(user_data_dir + "/" + input_json['uid'] + ".data", 'a') as userFile:
        userFile.write(str(input_json['timeToRate']) + "\t" + str(input_json['docNumber']) + "\t" + str(input_json['rating']) + "\n")
    return 'OK'

if __name__ == '__main__':
    get_dataset()
    app.run(debug=True,
            host='0.0.0.0',
            port=3000)
