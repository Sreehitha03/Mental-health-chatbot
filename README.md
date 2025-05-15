## Mental Health Chatbot – LLM-based Web Application
This is a web-based mental health chatbot that leverages a fine-tuned Large Language Model (LLM) to provide supportive conversations. The model is trained/fine-tuned in a Colab environment, stored on Google Drive, and served via a backend running on Colab (exposed using ngrok), while the frontend runs locally or on a hosting platform.

## 🔧 Features
Fine-tuned gemma-2b-4bit model using Unsloth.

Flask-based backend hosted on Google Colab

Accessible backend running on Google Colab with ngrok.

Chat history maintained across prompts.

Drive integration to store and load fine-tuned models.

## 🚀 Project Workflow
### 1. Training & Fine-Tuning
Model: unsloth/gemma-2b-4bit

Fine-tune using your dataset on Google Colab or a compatible GPU system.

After training, save the fine-tuned model to your Google Drive.

### 2. 📦 Backend (Flask + Transformers + Ngrok)
   
Clone the backend script in Colab.

Mount your Google Drive to access the trained model.

Start the Flask server and expose it using ngrok

You'll get a public ngrok URL like:
Running on http://xxxx.ngrok.io
Replace this in the fetch call.

### 3. 🌐 Frontend
Technologies Used
HTML, CSS, JS, React 

## 🛠️ Notes
If using Colab, ensure you re-authenticate pyngrok every time the runtime resets.

Keep your Google Drive mounted in Colab to load the model each time.

You can shift to EC2 for long-term backend hosting.

