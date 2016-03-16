"""Runs a user interface for the interactive anchor words algorithm"""

import flask
import os
import uuid
import threading
import pickle


app = flask.Flask(__name__, static_url_path='')


FILENAME = '/aml/data/amazon/amazon.txt'

# This is the number of documents each user is required to complete
REQUIRED_DOCS = 60


# Everything in this block needs to be run at server startup
# user_dict holds information on users
user_dict = {}
lock = threading.Lock()
# filedict is a docnumber to document dictionary
filedict = {}
# Here we populate filedict with docnumber as the key and document as the value
with open(FILENAME, 'r') as f:
    for line in f:
        filedict[line.split('\t')[0]] = line.split('\t')[1]
doc_order = []
# This holds where we currently are in doc_order
doc_order_index = 0
# Here we get the order of documents to be served
with open('best_order.pickle', 'rb') as f:
    d = pickle.load(f)
    for tup in d:
        doc_order.append(tup[0])
# This maintains state if the server crashes
try:
    last_state = open('last_state.pickle', 'rb')
except IOError:
    print('No last_state.pickle file, assuming no previous state')
else:
    state = pickle.load(last_state)
    with lock:
        user_dict = state['user_dict']
    doc_order_index = state['doc_order_index']
    print("Last state: " + str(state))
    last_state.close()


def save_state():
    """Saves the state of the server to a pickle file"""
    last_state = {}
    global user_dict
    last_state['user_dict'] = user_dict
    global doc_order_index
    last_state['doc_order_index'] = doc_order_index
    pickle.dump(last_state, open('last_state.pickle', 'wb'))

@app.route('/get_doc')
def get_doc():
    """Gets the next document for whoever is asking"""
    # If they've done all required documents but one, remove the id from
    #   the userDict and send the last document
    user_id = flask.request.headers.get('uuid')
    doc_number = 0
    document = ''
    completed = 0
    with lock:
        if user_id in user_dict:
            if user_dict[user_id]['num_docs'] == REQUIRED_DOCS:
                del user_dict[user_id]

    # Update the user_dict (unless this was the last document and the user_id
    #   was removed above)
    with lock:
        if user_id in user_dict:
            global doc_order_index
            doc_number = doc_order[doc_order_index]
            doc_order_index += 1
            document = filedict[doc_number]
            user_dict[user_id]['num_docs'] = user_dict[user_id]['num_docs'] + 1
            completed = user_dict[user_id]['num_docs']
            user_dict[user_id]['doc_number'] = doc_number
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
    doc_number = 0
    user_id = flask.request.headers.get('uuid')
    # Get info from the user_dict to use to get the old document
    with lock:
        if user_id in user_dict:
            doc_number = user_dict[user_id]['doc_number']
    # Return the document and doc_index to the client
    return flask.jsonify(document=filedict[doc_number], doc_number=doc_number)

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
    with lock:
        user_dict[str(uid)] = {'num_docs':0, 'doc_index':0}
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
