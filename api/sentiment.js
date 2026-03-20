const DEFAULT_MODEL = process.env.HF_MODEL || "distilbert-base-uncased-finetuned-sst-2-english";
const NEUTRAL_THRESHOLD = Number(process.env.NEUTRAL_THRESHOLD || 0.7);
const ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models";
const Sentiment = require("sentiment");
const sentimentEngine = new Sentiment();

const STRONG_NEGATIVE_TERMS = [
  "worst", "refund", "failing", "failed", "broke", "broken", "confusing",
  "hard to navigate", "terrible", "disappointed", "delay", "delayed", "issue"
];

const STRONG_POSITIVE_TERMS = [
  "exceeded", "fantastic", "highly recommended", "solved", "quickly", "delicious",
  "amazing", "great", "love", "thrilled"
];

const NEGATED_POSITIVE_PATTERNS = [
  "not good", "not great", "not happy", "not satisfied", "not amazing"
];

const NEGATED_NEGATIVE_PATTERNS = [
  "not bad", "not terrible", "not worst", "not awful"
];

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

function cleanToken(tokenValue) {
  let token = String(tokenValue || "").trim();
  if (token.toLowerCase().startsWith("bearer ")) {
    token = token.slice(7).trim();
  }
  if ((token.startsWith('"') && token.endsWith('"')) || (token.startsWith("'") && token.endsWith("'"))) {
    token = token.slice(1, -1).trim();
  }
  return token;
}

function summarizeHfError(response, data) {
  const status = response?.status || 0;
  const rawError = String(data?.error || "");
  const lower = rawError.toLowerCase();

  if (status === 401 || lower.includes("unauthorized") || lower.includes("credentials")) {
    return "Unauthorized (401) from Hugging Face Router. Check HUGGINGFACE_API_TOKEN value, scope, and environment.";
  }

  if (status === 403 || lower.includes("insufficient permissions")) {
    return "Forbidden (403) from Hugging Face Router. Token exists but lacks required permissions.";
  }

  if (lower.includes("rate limit") || status === 429) {
    return "Rate limited by Hugging Face Router (429). Try again later or use a higher quota token.";
  }

  if (lower.includes("<!doctype html") || lower.includes("<html")) {
    return `HTTP ${status}: Received HTML error page from Hugging Face Router.`;
  }

  return `HTTP ${status}: ${rawError || "Unknown inference error"}`;
}

function buildModelPath(modelId) {
  // Keep namespace/model path shape expected by router APIs.
  return String(modelId || "")
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
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
  const rawText = String(text || "");
  const lower = rawText.toLowerCase();

  // Base lexicon score from sentiment package.
  const result = sentimentEngine.analyze(rawText);
  let score = Number(result.comparative || 0) * 3.2;

  // In mixed sentences, sentiment after contrast words is often dominant.
  const contrastParts = lower.split(/\b(?:but|however|though|although|yet)\b/g);
  if (contrastParts.length > 1) {
    const tail = contrastParts[contrastParts.length - 1].trim();
    if (tail) {
      const tailScore = Number(sentimentEngine.analyze(tail).comparative || 0) * 4.2;
      score = score * 0.35 + tailScore * 0.65;
    }
  }

  // Phrase-level corrections for common review patterns.
  for (const term of STRONG_NEGATIVE_TERMS) {
    if (lower.includes(term)) {
      score -= 1.15;
    }
  }

  for (const term of STRONG_POSITIVE_TERMS) {
    if (lower.includes(term)) {
      score += 0.95;
    }
  }

  for (const phrase of NEGATED_POSITIVE_PATTERNS) {
    if (lower.includes(phrase)) {
      score -= 1.0;
    }
  }

  for (const phrase of NEGATED_NEGATIVE_PATTERNS) {
    if (lower.includes(phrase)) {
      score += 0.75;
    }
  }

  // Force Neutral for explicitly balanced phrases.
  if (lower.includes("nothing special") || lower.includes("average overall")) {
    score *= 0.35;
  }

  const magnitude = Math.min(1.0, Math.abs(score) / 3.5);

  if (Math.abs(score) < 0.42) {
    return {
      sentiment: "Neutral",
      confidence: Number((58 + magnitude * 20).toFixed(2))
    };
  }

  if (score > 0) {
    return {
      sentiment: "Positive",
      confidence: Number((64 + magnitude * 31).toFixed(2))
    };
  }

  return {
    sentiment: "Negative",
    confidence: Number((64 + magnitude * 31).toFixed(2))
  };
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

function normalizeLabel(label) {
  const mapped = mapLabel(label);
  const low = String(mapped || "").toLowerCase();
  if (low.includes("positive")) {
    return "Positive";
  }
  if (low.includes("negative")) {
    return "Negative";
  }
  if (low.includes("neutral")) {
    return "Neutral";
  }
  return "Neutral";
}

function toAppStyleSentiment(prediction) {
  const rawLabel = normalizeLabel(prediction?.label);
  const score = Number(prediction?.score || 0);

  // Keep parity with local app.py behavior for binary models.
  if ((rawLabel === "Positive" || rawLabel === "Negative") && score < NEUTRAL_THRESHOLD) {
    return {
      sentiment: "Neutral",
      confidence: Number(((1 - score) * 100).toFixed(2))
    };
  }

  return {
    sentiment: rawLabel,
    confidence: Number((score * 100).toFixed(2))
  };
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

    const token = cleanToken(
      process.env.HUGGINGFACE_API_TOKEN || process.env.HF_TOKEN || process.env.HUGGINGFACEHUB_API_TOKEN || ""
    );

    const encodedModel = buildModelPath(DEFAULT_MODEL);
    const candidateUrls = [`${ROUTER_BASE_URL}/${encodedModel}`];

    let response;
    let data;
    let lastErrorText = "";

    for (const url of candidateUrls) {
      // Attempt with token first, then anonymous (public model path).
      let result = await requestInference(url, text, token);
      response = result.response;
      data = result.data;
      lastErrorText = summarizeHfError(response, data);

      const permissionError = isAuthLikeError(response, data);

      // Some token types cannot call provider-routed inference. Retry once
      // anonymously for public models before failing.
      if (!response.ok && token && permissionError) {
        result = await requestInference(url, text, "");
        response = result.response;
        data = result.data;
        lastErrorText = summarizeHfError(response, data);
      }

      if (response.ok) {
        break;
      }
    }

    if (!response || !response.ok) {
      const fallback = localFallbackSentiment(text);
      return res.status(200).json({
        sentiment: fallback.sentiment,
        confidence: fallback.confidence,
        model: "sentiment-js-fallback",
        source: "local-fallback",
        reason: lastErrorText || "No successful response from inference endpoints"
      });
    }

    const prediction = bestPrediction(data);
    const normalized = toAppStyleSentiment(prediction);

    return res.status(200).json({
      sentiment: normalized.sentiment,
      confidence: normalized.confidence,
      model: DEFAULT_MODEL,
      source: "huggingface"
    });
  } catch (error) {
    return res.status(500).json({
      error: "Unexpected server error",
      details: error.message
    });
  }
};
