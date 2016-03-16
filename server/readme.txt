Remember to change FILENAME, ENGL_STOP, and LABELS to your local data in server.py. We should probably eventually use some kind of data prefix like Ankura now does.

Delete last_state.pickle to reset the state of the server (to get it serving documents from the first one in the list again).

If you need to reset a browser's state related to this study (get rid of the cookie that holds the last received document and uuid), navigate the browser to "<hostname>:3000/end.html".
