from flask import Flask, render_template, request, redirect, url_for
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import os

app = Flask(__name__)

# Define paths for saving texts
BOT_FOLDER = 'bot'
HUMAN_FOLDER = 'human'
os.makedirs(BOT_FOLDER, exist_ok=True)
os.makedirs(HUMAN_FOLDER, exist_ok=True)

class CNNLSTM(nn.Module):
    def __init__(self, bert_model, hidden_size=1024, num_layers=3, dropout=0.2):
        super(CNNLSTM, self).__init__()
        self.bert = bert_model
        embedding_size = self.bert.config.hidden_size
        self.conv1 = nn.Conv1d(in_channels=embedding_size, out_channels=512, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=1024, out_channels=2048, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 2)

    def forward(self, input_ids, attention_masks):
        with torch.no_grad():
            outputs = self.bert(input_ids, attention_mask=attention_masks)
            last_hidden_state = outputs.last_hidden_state

        # Permute dimensions for CNN
        last_hidden_state = last_hidden_state.permute(0, 2, 1)
        conv1_out = self.pool(torch.relu(self.conv1(last_hidden_state)))
        conv2_out = self.pool(torch.relu(self.conv2(conv1_out)))
        conv3_out = self.pool(torch.relu(self.conv3(conv2_out)))
        lstm_input = conv3_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.mean(dim=1)
        output = self.fc(lstm_out)
        return output
    

# Load pre-trained BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('model_weights_cnn_lstm')

# Load your trained model
bert_model = BertModel.from_pretrained('bert-large-uncased')
model = CNNLSTM(bert_model)
model.load_state_dict(torch.load('model_weights_cnn_lstm.pth', map_location=torch.device('cpu')))
model.eval()

def tokenize_text(text, max_length=128):
    encoded_dict = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    return encoded_dict['input_ids'], encoded_dict['attention_mask']

def predict_text(text):
    input_ids, attention_mask = tokenize_text(text)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        predictions = torch.argmax(outputs, dim=1).item()
    return predictions

def save_text(text, label):
    folder = HUMAN_FOLDER if label == 0 else BOT_FOLDER
    files = os.listdir(folder)
    file_count = len(files) + 1
    file_path = os.path.join(folder, f"{file_count}.txt")
    with open(file_path, 'w') as f:
        f.write(text)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        prediction = predict_text(text)
        save_text(text, prediction)
        return redirect(url_for('results', prediction=prediction))

@app.route('/results/<int:prediction>')
def results(prediction):
    label = 'Human' if prediction == 0 else 'Bot'
    return render_template('results.html', prediction=label)

@app.route('/images.html')
def show_images():
    return render_template('images.html')

if __name__ == '__main__':
    app.run(debug=True , port=5001)
