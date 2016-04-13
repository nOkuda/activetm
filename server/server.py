"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import os
import uuid
import threading
import pickle
import random


APP = flask.Flask(__name__, static_url_path='')
# users must label REQUIRED_DOCS documents
REQUIRED_DOCS = 100
ORDER_PICKLE = 'order.pickle'

def get_user_dict_on_start():
    """Loads user data"""
    # This maintains state if the server crashes
    try:
        last_state = open('last_state.pickle', 'rb')
    except IOError:
        print('No last_state.pickle file, assuming no previous state')
    else:
        state = pickle.load(last_state)
        print("Last state: " + str(state))
        last_state.close()
        return state['USER_DICT']
    # but if the server is starting fresh, so does the user data
    return {}

def get_filedict():
    """Loads FILEDICT"""
    with open('filedict.pickle', 'rb') as ifh:
        return pickle.load(ifh)

def get_doc_order():
    """Loads document order"""
    result = []
    # Here we get the order of documents to be served
    with open(ORDER_PICKLE, 'rb') as ifh:
        data = pickle.load(ifh)
    for tup in data:
        result.append(tup[0])
    return result

################################################################################
# Everything in this bLOCK needs to be run at server startup
# USER_DICT holds information on users
USER_DICT = get_user_dict_on_start()
LOCK = threading.Lock()
RNG = random.Random()
# FILEDICT is a docnumber to document dictionary
FILEDICT = get_filedict()
DOC_ORDER = get_doc_order()
################################################################################


def save_state():
    """Saves the state of the server to a pickle file"""
    last_state = {}
    last_state['USER_DICT'] = USER_DICT
    print(USER_DICT)
    pickle.dump(last_state, open('last_state.pickle', 'wb'))

@APP.route('/get_doc')
def get_doc():
    """Gets the next document for whoever is asking"""
    user_id = flask.request.headers.get('uuid')
    doc_number = 0
    document = ''
    with LOCK:
        if user_id in USER_DICT:
            completed = USER_DICT[user_id]['num_docs']
            if completed < REQUIRED_DOCS:
                doc_number = DOC_ORDER[
                    USER_DICT[user_id]['start_doc_index'] + completed]
                document = FILEDICT[doc_number]['text']
    # Return the document
    return flask.jsonify(document=document, doc_number=doc_number)

@APP.route('/old_doc')
def get_old_doc():
    """Gets the old document for someone if they have one,
        if they refreshed the page for instance"""
    # Create needed variables
    user_id = flask.request.headers.get('uuid')
    doc_number = 0
    completed = 0
    correct = 0
    document = ''
    # Get info from the USER_DICT to use to get the old document
    if user_id in USER_DICT:
        completed = USER_DICT[user_id]['num_docs']
        if completed < REQUIRED_DOCS:
            doc_number = USER_DICT[user_id]['doc_number']
            document = FILEDICT[doc_number]['text']
            completed = USER_DICT[user_id]['num_docs']
            correct = USER_DICT[user_id]['correct']
    # Return the document and doc_number to the client
    return flask.jsonify(
        document=document, doc_number=doc_number, completed=completed,
        correct=correct)

@APP.route('/')
def serve_landing_page():
    """Serves the landing page for the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'index.html')

@APP.route('/docs.html')
def serve_ui():
    """Serves the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'docs.html')

@APP.route('/scripts/script.js')
def serve_ui_js():
    """Serves the Javascript for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'script.js')

@APP.route('/end.html')
def serve_end_page():
    """Serves the end page for the Active TM UI"""
    return flask.send_from_directory('static', 'end.html')

@APP.route('/finalize')
def finalize():
    """Serves final statistics for the given user and erases the user from the
    database"""
    user_id = flask.request.headers.get('uuid')
    correct = 0
    complete = 0
    with LOCK:
        if user_id in USER_DICT:
            correct = USER_DICT[user_id]['correct']
            complete = USER_DICT[user_id]['num_docs']
            del USER_DICT[user_id]
            save_state()
    return flask.jsonify(correct=correct, complete=complete)

@APP.route('/scripts/end.js')
def serve_end_js():
    """Serves the Javascript for the end page for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'end.js')

@APP.route('/scripts/js.cookie.js')
def serve_cookie_script():
    """Serves the Javascript cookie script"""
    return flask.send_from_directory('static/scripts', 'js.cookie.js')

@APP.route('/stylesheets/style.css')
def serve_ui_css():
    """Serves the CSS file for the Active TM UI"""
    return flask.send_from_directory('static/stylesheets', 'style.css')

@APP.route('/uuid')
def get_uid():
    """Sends a UUID to the client"""
    uid = uuid.uuid4()
    data = {'id': uid}
    start_doc_index = RNG.randint(0, len(FILEDICT)-REQUIRED_DOCS)
    with LOCK:
        USER_DICT[str(uid)] = {
            'num_docs': 0,
            'start_doc_index': start_doc_index,
            'doc_number': DOC_ORDER[start_doc_index],
            'correct': 0}
    save_state()
    return flask.jsonify(data)

@APP.route('/rated', methods=['POST'])
def get_rating():
    """Receives and saves a user rating to a specific user's file"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    user_id = input_json['uid']
    doc_number = input_json['doc_number']
    guess = input_json['rating']
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    file_to_open = user_data_dir+"/"+user_id+".data"
    with open(file_to_open, 'a') as user_file:
        user_file.write(
            str(input_json['start_time']) + "\t" + str(input_json['end_time']) +
            "\t" + str(doc_number) + "\t" + str(guess) + "\n")
    prevlabel = FILEDICT[doc_number]['label']
    correct = USER_DICT[user_id]['correct']
    completed = 0
    with LOCK:
        if user_id in USER_DICT:
            if guess == prevlabel:
                USER_DICT[user_id]['correct'] += 1
                correct = USER_DICT[user_id]['correct']
            USER_DICT[user_id]['num_docs'] += 1
            completed = USER_DICT[user_id]['num_docs']
            if completed < REQUIRED_DOCS:
                next_doc_number = DOC_ORDER[
                    USER_DICT[user_id]['start_doc_index'] + completed]
                # advance index for next time
                USER_DICT[user_id]['doc_number'] = next_doc_number
    # Save state (in case the server crashes)
    save_state()
    return flask.jsonify(
        label=prevlabel, completed=completed, correct=correct)

if __name__ == '__main__':
    APP.run(debug=True,
            host='0.0.0.0',
            port=3000)

