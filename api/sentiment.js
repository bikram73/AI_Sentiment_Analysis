const DEFAULT_MODEL = process.env.HF_MODEL || "cardiffnlp/twitter-roberta-base-sentiment-latest";
const ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models";
const LEGACY_BASE_URL = "https://api-inference.huggingface.co/models";

async function requestInference(url, text, token) {
  const headers = {
    "Content-Type": "application/json"
  };

  if (token) {
    headers.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      inputs: text,
      options: {
        wait_for_model: true
      }
    })
  });

  const raw = await response.text();
  let data;
  try {
    data = raw ? JSON.parse(raw) : {};
  } catch {
    data = { error: raw || "Invalid response from inference API" };
  }

  return { response, data };
}

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

    const token = process.env.HUGGINGFACE_API_TOKEN || "";

    const encodedModel = encodeURIComponent(DEFAULT_MODEL);
    const candidateUrls = [
      `${ROUTER_BASE_URL}/${encodedModel}`,
      `${LEGACY_BASE_URL}/${encodedModel}`
    ];

    let response;
    let data;

    for (const url of candidateUrls) {
      let result = await requestInference(url, text, token);
      response = result.response;
      data = result.data;

      const errText = String(data?.error || "").toLowerCase();
      const permissionError = errText.includes("does not have sufficient permissions") || errText.includes("insufficient permissions");

      // Some token types cannot call provider-routed inference. Retry once
      // anonymously for public models before failing.
      if (!response.ok && token && permissionError) {
        result = await requestInference(url, text, "");
        response = result.response;
        data = result.data;
      }

      if (response.ok) {
        break;
      }

      const canRetryLegacy = errText.includes("no longer supported") || errText.includes("router.huggingface.co");
      if (!canRetryLegacy) {
        break;
      }
    }

    if (!response || !response.ok) {
      return res.status(response?.status || 502).json({
        error: data?.error || "Inference request failed",
        hint: "Use a Hugging Face token with Read permission or remove HUGGINGFACE_API_TOKEN to use anonymous access for public models."
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
