from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

model_name = "Salesforce/codegen2-7B_P"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

@app.route('/generate_code', methods=['POST'])
def generate_code():
    prompt = request.json.get('prompt')
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=256)  # Adjust max_length as needed
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_code': generated_code})

if __name__ == '__main__':
    app.run()

