from flask import Flask, request, jsonify
from utils import ask_question

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.post('/query')
def data_query():
    print("In test query req")
    query = request.json['query']

    print("In test query req: Query - ", query)
    response = ask_question(query)

    response_data = {
        "query" : query,
        "response" : response
    }

    return jsonify(response_data)

if __name__ == "__main__":
    app.run(debug=True, port=5005)