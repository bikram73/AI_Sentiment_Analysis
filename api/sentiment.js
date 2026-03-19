const DEFAULT_MODEL = process.env.HF_MODEL || "cardiffnlp/twitter-roberta-base-sentiment-latest";

function toTitle(text) {
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

function mapLabel(label) {
  const raw = String(label || "").toLowerCase();

  const labelMap = {
    negative: "Negative",
    neutral: "Neutral",
    positive: "Positive",
    label_0: "Negative",
    label_1: "Neutral",
    label_2: "Positive"
  };

  return labelMap[raw] || toTitle(raw || "Neutral");
}

function bestPrediction(result) {
  if (Array.isArray(result) && Array.isArray(result[0])) {
    return result[0].reduce((best, curr) => (curr.score > best.score ? curr : best), result[0][0]);
  }

  if (Array.isArray(result) && result.length > 0) {
    return result[0];
  }

  return { label: "neutral", score: 0.0 };
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const text = String(req.body?.text || "").trim();

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    const headers = {
      "Content-Type": "application/json"
    };

    if (process.env.HUGGINGFACE_API_TOKEN) {
      headers.Authorization = `Bearer ${process.env.HUGGINGFACE_API_TOKEN}`;
    }

    const response = await fetch(
      `https://api-inference.huggingface.co/models/${encodeURIComponent(DEFAULT_MODEL)}`,
      {
        method: "POST",
        headers,
        body: JSON.stringify({
          inputs: text,
          options: {
            wait_for_model: true
          }
        })
      }
    );

    const data = await response.json();

    if (!response.ok) {
      return res.status(response.status).json({
        error: data?.error || "Inference request failed"
      });
    }

    const prediction = bestPrediction(data);
    const sentiment = mapLabel(prediction.label);
    const confidence = Number(prediction.score || 0) * 100;

    return res.status(200).json({
      sentiment,
      confidence: Number(confidence.toFixed(2)),
      model: DEFAULT_MODEL
    });
  } catch (error) {
    return res.status(500).json({
      error: "Unexpected server error",
      details: error.message
    });
  }
};
