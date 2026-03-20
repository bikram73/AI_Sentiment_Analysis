const DEFAULT_MODEL = process.env.HF_MODEL || "cardiffnlp/twitter-roberta-base-sentiment-latest";
const DEFAULT_MULTILINGUAL_MODEL = process.env.HF_MODEL_MULTILINGUAL || "cardiffnlp/twitter-xlm-roberta-base-sentiment";
const DEFAULT_EMOTION_MODEL = process.env.HF_MODEL_EMOTION || "j-hartmann/emotion-english-distilroberta-base";
const NEUTRAL_THRESHOLD = Number(process.env.NEUTRAL_THRESHOLD || 0.7);
const ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models";
const Sentiment = require("sentiment");
const sentimentEngine = new Sentiment();

const VALID_MODES = new Set(["sentiment", "emotion", "multilingual", "absa"]);
const MAX_BATCH_SIZE = 25;

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

async function requestInference(url, text, token, extraBody = {}) {
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
      },
      ...extraBody
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

function cleanModelId(modelValue) {
  let model = String(modelValue || "").trim();
  if ((model.startsWith('"') && model.endsWith('"')) || (model.startsWith("'") && model.endsWith("'"))) {
    model = model.slice(1, -1).trim();
  }
  return model;
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

function getRequestedMode(modeValue) {
  const mode = String(modeValue || "sentiment").trim().toLowerCase();
  return VALID_MODES.has(mode) ? mode : "sentiment";
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

function mapEmotionLabel(label) {
  const raw = String(label || "").trim().toLowerCase();
  const emotions = {
    joy: "Joy",
    anger: "Anger",
    sadness: "Sadness",
    fear: "Fear",
    surprise: "Surprise",
    disgust: "Disgust",
    neutral: "Neutral",
    love: "Love"
  };
  return emotions[raw] || toTitle(raw || "Neutral");
}

function flattenPredictions(result) {
  if (Array.isArray(result) && Array.isArray(result[0])) {
    return result[0];
  }

  if (Array.isArray(result)) {
    return result;
  }

  if (result && typeof result === "object" && "label" in result) {
    return [result];
  }

  return [];
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
  const predictions = flattenPredictions(result);
  if (predictions.length > 0) {
    return predictions.reduce((best, curr) => (curr.score > best.score ? curr : best), predictions[0]);
  }

  return { label: "neutral", score: 0.0 };
}

function emotionFromSentimentFallback(text) {
  const base = localFallbackSentiment(text);
  if (base.sentiment === "Positive") {
    return { emotion: "Joy", confidence: base.confidence };
  }
  if (base.sentiment === "Negative") {
    return { emotion: "Anger", confidence: base.confidence };
  }
  return { emotion: "Neutral", confidence: base.confidence };
}

function buildAbsaResult(text) {
  const rawText = String(text || "");
  const lower = rawText.toLowerCase();

  const aspectMap = {
    Food: ["food", "taste", "meal", "dish", "flavor"],
    Service: ["service", "staff", "waiter", "support", "helpdesk"],
    Delivery: ["delivery", "shipping", "arrived", "delay", "delayed"],
    Price: ["price", "cost", "expensive", "cheap", "value"],
    Quality: ["quality", "material", "build", "durable", "broken", "broke"],
    App: ["app", "interface", "ui", "checkout", "navigation", "performance"]
  };

  const clauses = rawText
    .split(/(?<=[.!?])\s+|\bbut\b|\bhowever\b|\bthough\b/gi)
    .map((part) => part.trim())
    .filter(Boolean);

  const aspects = [];

  for (const [aspect, keywords] of Object.entries(aspectMap)) {
    const hit = clauses.find((clause) => keywords.some((kw) => clause.toLowerCase().includes(kw)));
    if (!hit) {
      continue;
    }

    const local = localFallbackSentiment(hit);
    aspects.push({
      aspect,
      sentiment: local.sentiment,
      confidence: local.confidence,
      text: hit
    });
  }

  if (aspects.length === 0) {
    const local = localFallbackSentiment(rawText);
    return {
      overallSentiment: local.sentiment,
      overallConfidence: local.confidence,
      aspects: []
    };
  }

  const scoreMap = { Positive: 1, Neutral: 0, Negative: -1 };
  const weightedScore = aspects.reduce(
    (acc, item) => acc + (scoreMap[item.sentiment] || 0) * (item.confidence / 100),
    0
  );

  let overallSentiment = "Neutral";
  if (weightedScore > 0.25) {
    overallSentiment = "Positive";
  } else if (weightedScore < -0.25) {
    overallSentiment = "Negative";
  }

  const overallConfidence = Number(
    (aspects.reduce((sum, item) => sum + item.confidence, 0) / aspects.length).toFixed(2)
  );

  return {
    overallSentiment,
    overallConfidence,
    aspects
  };
}

function getCandidateModels(mode, configuredModel) {
  if (mode === "emotion") {
    return [
      cleanModelId(DEFAULT_EMOTION_MODEL),
      "j-hartmann/emotion-english-distilroberta-base"
    ].filter((value, index, arr) => Boolean(value) && arr.indexOf(value) === index);
  }

  if (mode === "multilingual") {
    return [
      cleanModelId(DEFAULT_MULTILINGUAL_MODEL),
      "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    ].filter((value, index, arr) => Boolean(value) && arr.indexOf(value) === index);
  }

  return [
    configuredModel,
    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "distilbert-base-uncased-finetuned-sst-2-english"
  ].filter((value, index, arr) => Boolean(value) && arr.indexOf(value) === index);
}

async function analyzeSingleText(text, mode, token, configuredModel) {
  if (mode === "absa") {
    const absa = buildAbsaResult(text);
    return {
      mode,
      sentiment: absa.overallSentiment,
      confidence: absa.overallConfidence,
      aspects: absa.aspects,
      model: "absa-heuristic",
      source: "local-absa"
    };
  }

  const candidateModels = getCandidateModels(mode, configuredModel);

  let response;
  let data;
  let lastErrorText = "";
  let selectedModel = configuredModel;

  for (const modelId of candidateModels) {
    const url = `${ROUTER_BASE_URL}/${buildModelPath(modelId)}`;
    selectedModel = modelId;

    // Attempt with token first, then anonymous (public model path).
    const extraBody = mode === "emotion" ? { parameters: { top_k: null } } : {};
    let result = await requestInference(url, text, token, extraBody);
    response = result.response;
    data = result.data;
    lastErrorText = summarizeHfError(response, data);

    const permissionError = isAuthLikeError(response, data);

    // Some token types cannot call provider-routed inference. Retry once
    // anonymously for public models before failing.
    if (!response.ok && token && permissionError) {
      result = await requestInference(url, text, "", extraBody);
      response = result.response;
      data = result.data;
      lastErrorText = summarizeHfError(response, data);
    }

    if (response.ok) {
      break;
    }

    // 404 often means wrong or unavailable model ID. Try next known model.
    if (response.status === 404) {
      continue;
    }

    // For non-404 failures, stop trying model variants.
    break;
  }

  if (!response || !response.ok) {
    if (mode === "emotion") {
      const fallbackEmotion = emotionFromSentimentFallback(text);
      return {
        mode,
        emotion: fallbackEmotion.emotion,
        confidence: fallbackEmotion.confidence,
        top_emotions: [{ label: fallbackEmotion.emotion, score: Number((fallbackEmotion.confidence / 100).toFixed(4)) }],
        model: "emotion-fallback",
        source: "local-fallback",
        reason: lastErrorText || "No successful response from inference endpoints"
      };
    }

    const fallback = localFallbackSentiment(text);
    return {
      mode,
      sentiment: fallback.sentiment,
      confidence: fallback.confidence,
      model: "sentiment-js-fallback",
      source: "local-fallback",
      reason: lastErrorText || "No successful response from inference endpoints"
    };
  }

  if (mode === "emotion") {
    const predictions = flattenPredictions(data)
      .map((item) => ({ label: mapEmotionLabel(item.label), score: Number(item.score || 0) }))
      .sort((a, b) => b.score - a.score);

    const top = predictions[0] || { label: "Neutral", score: 0 };
    return {
      mode,
      emotion: top.label,
      confidence: Number((top.score * 100).toFixed(2)),
      top_emotions: predictions.slice(0, 5),
      model: selectedModel,
      source: "huggingface"
    };
  }

  const prediction = bestPrediction(data);
  const normalized = toAppStyleSentiment(prediction);

  return {
    mode,
    sentiment: normalized.sentiment,
    confidence: normalized.confidence,
    model: selectedModel,
    source: "huggingface"
  };
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const text = String(req.body?.text || "").trim();
    const rawTexts = Array.isArray(req.body?.texts) ? req.body.texts : null;
    const mode = getRequestedMode(req.body?.mode);
    const batchTexts = rawTexts
      ? rawTexts
          .map((item) => String(item || "").trim())
          .filter(Boolean)
      : [];

    if (!text && batchTexts.length === 0) {
      return res.status(400).json({ error: "Text is required" });
    }

    if (batchTexts.length > MAX_BATCH_SIZE) {
      return res.status(400).json({
        error: `Batch size exceeded. Maximum ${MAX_BATCH_SIZE} items are allowed per request.`
      });
    }

    const token = cleanToken(
      process.env.HUGGINGFACE_API_TOKEN || process.env.HF_TOKEN || process.env.HUGGINGFACEHUB_API_TOKEN || ""
    );

    const configuredModel = cleanModelId(DEFAULT_MODEL);

    if (batchTexts.length > 0) {
      const results = await Promise.all(
        batchTexts.map(async (item, idx) => {
          const analyzed = await analyzeSingleText(item, mode, token, configuredModel);
          return {
            index: idx + 1,
            text: item,
            ...analyzed
          };
        })
      );

      const averageConfidence = Number(
        (results.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / results.length).toFixed(2)
      );

      return res.status(200).json({
        mode,
        batch: true,
        count: results.length,
        confidence: averageConfidence,
        results
      });
    }

    const singleResult = await analyzeSingleText(text, mode, token, configuredModel);
    return res.status(200).json(singleResult);
  } catch (error) {
    return res.status(500).json({
      error: "Unexpected server error",
      details: error.message
    });
  }
};
