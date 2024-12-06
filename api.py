from flask import Flask, request, jsonify
from flask_cors import CORS

from funcs.lex import *
from translate import *

app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_world():
    return "<p>Hello, World</p>"

@app.route("/translate", methods=['POST'])
def translate():
    data = request.get_json()
    code = data.get('code')
    code_type = data.get('code_type')

    if not code and not code_type:
        return jsonify({
            'error': 'Both atributte are necessary code and code_type'
        }), 400

    result_lex = detect_language(code, code_type)

    if result_lex["status"] == True:
        print(result_lex["status"])
        try:
            asm_code = translate_to_asm(code)
            
            return jsonify({
                'status' : True,
                'code' : asm_code
            }), 200
            
        except Exception as e:
            return jsonify({
                'error' : f"An Error have ocurred: {e}"
            }), 500
    else :
        return jsonify({
            'status' : False,
            'error' : result_lex['content']
        }), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
