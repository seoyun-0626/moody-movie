from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 허용 (다른 컴퓨터/브라우저에서 접속 가능하게)

@app.route('/send', methods=['POST'])
def receive_data():
    data = request.json
    print("받은 데이터:", data)
    return jsonify({'status': 'ok', 'received': data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)