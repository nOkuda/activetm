"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import os
import uuid
import threading
import pickle
import random


app = flask.Flask(__name__, static_url_path='')
# users must label REQUIRED_DOCS documents
REQUIRED_DOCS = 10
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
        return state['user_dict']
    # but if the server is starting fresh, so does the user data
    return {}

def get_filedict():
    """Loads filedict"""
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
# Everything in this block needs to be run at server startup
# user_dict holds information on users
user_dict = get_user_dict_on_start()
lock = threading.Lock()
rng = random.Random()
# filedict is a docnumber to document dictionary
filedict = get_filedict()
doc_order = get_doc_order()
################################################################################


def save_state():
    """Saves the state of the server to a pickle file"""
    last_state = {}
    global user_dict
    last_state['user_dict'] = user_dict
    print(user_dict)
    pickle.dump(last_state, open('last_state.pickle', 'wb'))

@app.route('/get_doc')
def get_doc():
    """Gets the next document for whoever is asking"""
    user_id = flask.request.headers.get('uuid')
    doc_number = 0
    document = ''
    completed = 0
    # Update the user_dict
    with lock:
        if user_id in user_dict:
            completed = user_dict[user_id]['num_docs']
            if completed < REQUIRED_DOCS:
                doc_number = doc_order[user_dict[user_id]['doc_index']]
                document = filedict[doc_number]
                user_dict[user_id]['doc_index'] += 1
                user_dict[user_id]['num_docs'] += 1
                user_dict[user_id]['doc_number'] = doc_number
            else:
                del user_dict[user_id]
    # Save state (in case the server crashes)
    save_state()
    # Return the document
    return flask.jsonify(
        document=document, doc_number=doc_number, completed=completed)


@app.route('/old_doc')
def get_old_doc():
    """Gets the old document for someone if they have one,
        if they refreshed the page for instance"""
    # Create needed variables
    user_id = flask.request.headers.get('uuid')
    doc_number = 0
    completed = 0
    # Get info from the user_dict to use to get the old document
    with lock:
        if user_id in user_dict:
            doc_number = user_dict[user_id]['doc_number']
            # num_docs tells how many different documents have been served, so
            # going back for an old document means one less has been completed
            completed = user_dict[user_id]['num_docs'] - 1
    # Return the document and doc_number to the client
    return flask.jsonify(
        document=filedict[doc_number], doc_number=doc_number,
        completed=completed)

@app.route('/')
def serve_landing_page():
    """Serves the landing page for the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'index.html')

@app.route('/docs.html')
def serve_ui():
    """Serves the Active Topic Modeling UI"""
    return flask.send_from_directory('static', 'docs.html')


@app.route('/scripts/script.js')
def serve_ui_js():
    """Serves the Javascript for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'script.js')


@app.route('/end.html')
def serve_end_page():
    """Serves the end page for the Active TM UI"""
    user_id = flask.request.headers.get('uuid')
    with lock:
        if user_id in user_dict:
            print(user_id+'navigated to end.html')
            del user_dict[user_id]
            save_state()
    return flask.send_from_directory('static', 'end.html')


@app.route('/scripts/end.js')
def serve_end_js():
    """Serves the Javascript for the end page for the Active TM UI"""
    return flask.send_from_directory('static/scripts', 'end.js')


@app.route('/scripts/js.cookie.js')
def serve_cookie_script():
    """Serves the Javascript cookie script"""
    return flask.send_from_directory('static/scripts', 'js.cookie.js')


@app.route('/stylesheets/style.css')
def serve_ui_css():
    """Serves the CSS file for the Active TM UI"""
    return flask.send_from_directory('static/stylesheets', 'style.css')


@app.route('/uuid')
def get_uid():
    """Sends a UUID to the client"""
    uid = uuid.uuid4()
    data = {'id': uid}
    doc_index = rng.randint(0, len(filedict)-REQUIRED_DOCS)
    with lock:
        user_dict[str(uid)] = {'num_docs':0, 'doc_index':doc_index}
    print(user_dict)
    return flask.jsonify(data)

@app.route('/rated', methods=['POST'])
def get_rating():
    """Receives and saves a user rating to a specific user's file"""
    flask.request.get_data()
    input_json = flask.request.get_json(force=True)
    user_data_dir = os.path.dirname(os.path.realpath(__file__)) + "/userData"
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir)
    file_to_open = user_data_dir+"/"+input_json['uid']+".data"
    with open(file_to_open, 'a') as user_file:
        user_file.write(str(input_json['start_time']) + "\t" + str(input_json['end_time']) + "\t" + str(input_json['doc_number']) + "\t" + str(input_json['rating']) + "\n")
    return 'OK'

if __name__ == '__main__':
    app.run(debug=True,
            host='0.0.0.0',
            port=3000)

