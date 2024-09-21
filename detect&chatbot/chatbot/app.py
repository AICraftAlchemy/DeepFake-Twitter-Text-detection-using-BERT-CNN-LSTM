from flask import Flask, render_template, request
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

# Load GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
gpt2 = pipeline("text-generation", model=model, tokenizer=tokenizer)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get-response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = gpt2(user_input, max_length=100)[0]['generated_text']
    return response

if __name__ == "__main__":
    app.run(debug=True, port=8080)
