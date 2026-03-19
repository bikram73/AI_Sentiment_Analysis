const DEFAULT_MODEL = process.env.HF_MODEL || "cardiffnlp/twitter-roberta-base-sentiment-latest";
const ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models";
const LEGACY_BASE_URL = "https://api-inference.huggingface.co/models";

const POSITIVE_WORDS = new Set([
  "amazing", "awesome", "best", "brilliant", "excellent", "fantastic", "good", "great", "happy",
  "impressive", "incredible", "love", "loved", "nice", "perfect", "recommend", "satisfied", "super"
]);

const NEGATIVE_WORDS = new Set([
  "awful", "bad", "broken", "disappointed", "error", "hate", "horrible", "issue", "poor",
  "refund", "sad", "slow", "terrible", "ugly", "unhappy", "worst", "fail", "failed"
]);

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

function isAuthLikeError(response, data) {
  const status = response?.status || 0;
  if (status === 401 || status === 403) {
    return true;
  }

  const errText = String(data?.error || "").toLowerCase();
  return (
    errText.includes("unauthorized") ||
    errText.includes("authorization") ||
    errText.includes("credentials") ||
    errText.includes("insufficient permissions") ||
    errText.includes("does not have sufficient permissions")
  );
}

function localFallbackSentiment(text) {
  const words = (text.toLowerCase().match(/[a-z']+/g) || []);

  let positive = 0;
  let negative = 0;

  for (const w of words) {
    if (POSITIVE_WORDS.has(w)) {
      positive += 1;
    }
    if (NEGATIVE_WORDS.has(w)) {
      negative += 1;
    }
  }

  const totalHits = positive + negative;

  if (totalHits === 0 || Math.abs(positive - negative) <= 1) {
    return { sentiment: "Neutral", confidence: 55.0 };
  }

  if (positive > negative) {
    const ratio = positive / totalHits;
    return { sentiment: "Positive", confidence: Number((60 + ratio * 35).toFixed(2)) };
  }

  const ratio = negative / totalHits;
  return { sentiment: "Negative", confidence: Number((60 + ratio * 35).toFixed(2)) };
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
      const permissionError = isAuthLikeError(response, data);

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

      const latestErr = String(data?.error || "").toLowerCase();
      const canRetryLegacy = latestErr.includes("no longer supported") || latestErr.includes("router.huggingface.co");
      if (!canRetryLegacy) {
        break;
      }
    }

    if (!response || !response.ok) {
      const fallback = localFallbackSentiment(text);
      return res.status(200).json({
        sentiment: fallback.sentiment,
        confidence: fallback.confidence,
        model: "local-fallback",
        warning: "Hugging Face inference unavailable. Returned local fallback sentiment."
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
