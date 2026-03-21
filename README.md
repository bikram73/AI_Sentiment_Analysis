# AI Sentiment Analysis Web App

Analyze any sentence or paragraph and predict sentiment as:

- Positive 😊
- Negative 😡
- Neutral 😐

This repository now supports two deployment paths:

- Hugging Face Spaces (Gradio app in Python)
- Vercel (web UI + serverless API using Hugging Face Inference API)

## ✨ Features

- Beautiful and responsive sentiment analysis interface
- Confidence score with visual progress meter
- 5 analysis modes: Sentiment, Emotion, Multilingual, ABSA, and Batch Input Analysis
- 15 curated examples per analysis mode
- Fast Vercel deployment using serverless endpoint
- Gradio app for local development and Hugging Face Spaces

## 🧰 Tech Stack

- Python 3.10+
- Gradio
- Hugging Face Transformers
- PyTorch
- Vercel Serverless Functions (Node.js)
- Hugging Face Inference API

## 🏗️ Architecture

Text Input -> Gradio Interface -> Transformers Pipeline -> Sentiment + Confidence Output

## ⚙️ Environment Variables

You can customize behavior using environment variables:

```bash
# Optional: only for local Gradio app
MODEL_NAME=distilbert-base-uncased-finetuned-sst-2-english
NEUTRAL_THRESHOLD=0.70
```

## 🚀 Run Locally (Gradio)

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open: `http://127.0.0.1:7860`

## ☁️ Deploy To Vercel

### 1. Push project to GitHub first

```bash
git init
git add .
git commit -m "Initial commit: AI sentiment app with Vercel support"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Import repository in Vercel

1. Login to Vercel.
2. Click Add New -> Project.
3. Import your GitHub repository.
4. Framework preset: Other.
5. Add environment variables in Vercel Project Settings:
	- `HUGGINGFACE_API_TOKEN`
	- `HF_MODEL` (optional)
6. Deploy.

Vercel uses:

- `index.html` for UI
- `api/sentiment.js` for inference API proxy
- `vercel.json` for routing

## 🤗 Deploy To Hugging Face Spaces

1. Create a new Space.
2. Select SDK: Gradio.
3. Upload these files:
	- `app.py`
	- `requirements.txt`
	- `README.md`
	- `.gitignore`
4. Wait for build completion.

## 📁 File Structure

```text
AI_Sentiment_Analysis_Web_App/
├── api/
│   └── sentiment.js          # Vercel serverless API route
├── app.py                    # Gradio app (Python)
├── index.html                # Home page project details
├── Sentimental_Analysis.html # AI Sentimental Analysis Predictor
├── vercel.json               # Vercel routing config
├── requirements.txt          # Python dependencies for Gradio app
├── .env.example              # Example environment variables
├── .gitignore                # Git ignore rules
└── README.md                 # Documentation
```

## 🧪 Example Inputs

- I love this product. It works really well and exceeded my expectations.
- The service was okay. Nothing special, but not terrible either.
- This is the worst experience I have had. I want a refund immediately.

The Vercel analyzer page includes 15 examples for each analysis mode, including Batch Input Analysis.

## ⏱️ First Run Download Note

On first launch, the model is downloaded and cached locally. This can take time depending on your internet speed. Later runs are much faster because the cached model is reused.

## 🔮 Future Improvements

- User history and export options
- Confidence trend charts
- Custom model selection per analysis mode

## License

MIT. See `LICENSE`.