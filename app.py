import os
from functools import lru_cache

import gradio as gr
from transformers import pipeline

# Faster default model (~250MB) for lower first-run download time.
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
# If confidence is below this threshold, treat output as Neutral.
NEUTRAL_THRESHOLD = float(os.getenv("NEUTRAL_THRESHOLD", "0.70"))

EXAMPLE_TEXTS = [
    "I love this product. It works really well and exceeded my expectations.",
    "The service was okay. Nothing special, but not terrible either.",
    "This is the worst experience I have had. I want a refund immediately.",
    "The app is clean and easy to use. I found everything quickly.",
    "Delivery was delayed by two days, but customer support handled it politely.",
    "I am very disappointed with the quality. It broke on the first day.",
    "The food tasted fresh and delicious. I will order again.",
    "It is acceptable for the price, though performance could be better.",
    "Absolutely fantastic experience from start to finish.",
    "I feel neutral about this update. It changed a few things but nothing major.",
    "The product page looked great, but checkout kept failing.",
    "Support responded quickly and solved my issue in minutes.",
    "The interface is confusing and hard to navigate.",
    "Not bad, not great, just average overall.",
    "I am thrilled with the results. Highly recommended.",
]

EXAMPLE_CHOICES = [f"{idx + 1}. {text}" for idx, text in enumerate(EXAMPLE_TEXTS)]


CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

:root {
    --bg-start: #f4f7fb;
    --bg-end: #e8f2ff;
    --panel: #ffffff;
    --text: #14213d;
    --muted: #5c6b82;
    --accent: #0f766e;
    --accent-soft: #ccfbf1;
    --danger: #c1121f;
    --warning: #b45309;
    --shadow: 0 16px 40px rgba(20, 33, 61, 0.12);
}

.gradio-container {
    font-family: 'Space Grotesk', sans-serif !important;
    background: radial-gradient(circle at 15% 15%, #dff6ff 0%, transparent 38%),
                            radial-gradient(circle at 85% 0%, #ffe8cc 0%, transparent 35%),
                            linear-gradient(160deg, var(--bg-start), var(--bg-end));
}

#hero {
    background: linear-gradient(135deg, #052e2b 0%, #0f766e 55%, #14b8a6 100%);
    color: #ffffff;
    border-radius: 24px;
    padding: 28px;
    box-shadow: var(--shadow);
}

#hero h1 {
    margin: 0;
    font-size: 2rem;
    letter-spacing: 0.2px;
}

#hero p {
    margin: 10px 0 0;
    color: #d1fae5;
    line-height: 1.45;
}

.panel {
    background: var(--panel);
    border-radius: 20px;
    box-shadow: var(--shadow);
    border: 1px solid #dbe7f7;
}

#result-shell {
    min-height: 190px;
    display: flex;
    align-items: stretch;
}

.result-card {
    width: 100%;
    border-radius: 18px;
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #d8e6ff;
    padding: 18px;
    animation: reveal 260ms ease-out;
}

@keyframes reveal {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.pill {
    padding: 6px 12px;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
}

.pill.positive { background: #dcfce7; color: #166534; }
.pill.negative { background: #fee2e2; color: #991b1b; }
.pill.neutral { background: #fef3c7; color: #92400e; }

.result-label {
    margin: 0;
    color: var(--muted);
    font-size: 0.9rem;
}

.result-value {
    margin: 6px 0 0;
    color: var(--text);
    font-size: 1.35rem;
    font-weight: 700;
}

.meter {
    margin-top: 14px;
    height: 10px;
    width: 100%;
    border-radius: 999px;
    background: #e8eef9;
    overflow: hidden;
}

.meter > span {
    display: block;
    height: 100%;
    background: linear-gradient(90deg, #0ea5e9 0%, #0f766e 100%);
}

.helper {
    margin-top: 10px;
    color: var(--muted);
    font-size: 0.92rem;
}

@media (max-width: 768px) {
    #hero h1 { font-size: 1.6rem; }
}
"""


@lru_cache(maxsize=1)
def get_classifier():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return pipeline("sentiment-analysis", model=MODEL_NAME)


def normalize_label(raw_label: str) -> str:
    label = raw_label.strip().lower()

    label_map = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "label_0": "Negative",
        "label_1": "Neutral",
        "label_2": "Positive",
    }

    return label_map.get(label, raw_label.title())


def build_result_card(sentiment: str, confidence_value: float) -> str:
    sentiment_class = sentiment.lower()
    emoji_map = {"positive": "😊", "negative": "😡", "neutral": "😐"}
    emoji = emoji_map.get(sentiment_class, "🙂")
    confidence_pct = max(0.0, min(confidence_value, 100.0))

    return f"""
    <div class=\"result-card\">
        <div class=\"result-top\">
            <div>
                <p class=\"result-label\">Predicted Sentiment</p>
                <p class=\"result-value\">{emoji} {sentiment}</p>
            </div>
            <span class=\"pill {sentiment_class}\">{confidence_pct:.2f}% confidence</span>
        </div>
        <div class=\"meter\"><span style=\"width:{confidence_pct:.2f}%\"></span></div>
        <p class=\"helper\">Tip: Very mixed statements can appear as Neutral.</p>
    </div>
    """


def load_example(selection: str) -> str:
    if not selection:
        return ""

    try:
        index = int(selection.split(".", 1)[0]) - 1
    except (ValueError, IndexError):
        return ""

    if 0 <= index < len(EXAMPLE_TEXTS):
        return EXAMPLE_TEXTS[index]
    return ""


def analyze_sentiment(text: str):
    if not text or not text.strip():
        empty_card = """
        <div class=\"result-card\">
            <p class=\"result-label\">Waiting For Input</p>
            <p class=\"result-value\">Paste text and click Analyze Sentiment.</p>
            <p class=\"helper\">You can enter product reviews, tweets, comments, or feedback.</p>
        </div>
        """
        return "Please enter some text to analyze.", "N/A", empty_card

    classifier = get_classifier()
    result = classifier(text, truncation=True)[0]

    raw_label = normalize_label(result["label"])
    score = float(result["score"])

    # DistilBERT is binary (Positive/Negative). We expose Neutral when confidence
    # is low to preserve the 3-class UX while keeping startup fast.
    if score < NEUTRAL_THRESHOLD:
        sentiment = "Neutral"
        confidence_value = (1.0 - score) * 100
    else:
        sentiment = raw_label
        confidence_value = score * 100

    confidence = f"{confidence_value:.2f}%"
    card_html = build_result_card(sentiment, confidence_value)

    return sentiment, confidence, card_html


with gr.Blocks(title="AI Sentiment Analyzer", css=CUSTOM_CSS) as demo:
    gr.HTML(
        """
        <section id=\"hero\">
          <h1>AI Sentiment Analyzer</h1>
          <p>Discover emotional tone in seconds. Enter a sentence or full paragraph to classify it as Positive, Negative, or Neutral.</p>
        </section>
        """
    )

    with gr.Row():
        with gr.Column(scale=7, elem_classes=["panel"]):
            text_input = gr.Textbox(
                lines=8,
                label="Input Text",
                placeholder="Type your review, feedback, or message here...",
                container=True,
            )

            with gr.Row():
                analyze_button = gr.Button("Analyze Sentiment", variant="primary")
                clear_button = gr.Button("Clear")

            with gr.Accordion("Example Inputs ▼", open=False):
                gr.Markdown("Choose any example from the down-arrow list, then click Load Example.")
                example_dropdown = gr.Dropdown(
                    choices=EXAMPLE_CHOICES,
                    label="Select Example",
                    value=None,
                )
                load_example_button = gr.Button("Load Example ▼")

        with gr.Column(scale=5, elem_classes=["panel"], elem_id="result-shell"):
            result_card = gr.HTML(
                """
                <div class=\"result-card\">
                  <p class=\"result-label\">Ready</p>
                  <p class=\"result-value\">Your analysis result will appear here.</p>
                </div>
                """
            )
            sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)

    analyze_button.click(
        fn=analyze_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output, result_card],
    )

    clear_button.click(
        fn=lambda: (
            "",
            "",
            "",
            """
            <div class=\"result-card\">
              <p class=\"result-label\">Ready</p>
              <p class=\"result-value\">Your analysis result will appear here.</p>
            </div>
            """,
        ),
        inputs=None,
        outputs=[text_input, sentiment_output, confidence_output, result_card],
    )

    load_example_button.click(
        fn=load_example,
        inputs=example_dropdown,
        outputs=text_input,
    )


if __name__ == "__main__":
    demo.launch()
