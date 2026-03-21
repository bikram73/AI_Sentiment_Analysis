const DEFAULT_MODEL = process.env.HF_MODEL || "cardiffnlp/twitter-roberta-base-sentiment-latest";
const DEFAULT_MULTILINGUAL_MODEL = process.env.HF_MODEL_MULTILINGUAL || "cardiffnlp/twitter-xlm-roberta-base-sentiment";
const DEFAULT_EMOTION_MODEL = process.env.HF_MODEL_EMOTION || "j-hartmann/emotion-english-distilroberta-base";
const NEUTRAL_THRESHOLD = Number(process.env.NEUTRAL_THRESHOLD || 0.7);
const ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models";
const Sentiment = require("sentiment");
const sentimentEngine = new Sentiment();

const VALID_MODES = new Set(["sentiment", "emotion", "multilingual", "absa", "batch"]);

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

const LANGUAGE_NAME_MAP = {
  en: "English",
  es: "Spanish",
  fr: "French",
  de: "German",
  it: "Italian",
  pt: "Portuguese",
  hi: "Hindi",
  ar: "Arabic",
  ja: "Japanese",
  ko: "Korean",
  zh: "Chinese",
  ru: "Russian",
  tr: "Turkish",
  nl: "Dutch",
  ro: "Romanian",
  pl: "Polish",
  bn: "Bengali",
  kn: "Kannada",
  unknown: "Unknown"
};

const LATIN_LANGUAGE_HINTS = {
  es: [" el ", " la ", " que ", " de ", " no ", " muy ", " este ", " esta ", " excelente ", " producto "],
  fr: [" le ", " la ", " que ", " de ", " pas ", " tres ", " service ", " produit "],
  de: [" der ", " die ", " und ", " nicht ", " sehr ", " das ", " ist ", " gut ", " schlecht "],
  it: [" il ", " la ", " che ", " non ", " molto ", " questo ", " servizio ", " prodotto "],
  pt: [" o ", " a ", " que ", " nao ", " muito ", " este ", " produto ", " servico "],
  tr: [" ve ", " bu ", " cok ", " degil ", " hizmet ", " urun "],
  nl: [" de ", " het ", " en ", " niet ", " erg ", " service ", " product "],
  ro: [" si ", " este ", " foarte ", " nu ", " produs ", " serviciu "],
  pl: [" i ", " jest ", " bardzo ", " nie ", " produkt ", " usluga "]
};

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

function splitBatchInputs(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .slice(0, 50);
}

function aggregateBatch(items) {
  if (!Array.isArray(items) || items.length === 0) {
    return {
      sentiment: "Neutral",
      confidence: 0
    };
  }

  const scoreMap = { Positive: 1, Neutral: 0, Negative: -1 };
  const score = items.reduce(
    (acc, item) => acc + (scoreMap[item.sentiment] || 0) * (Number(item.confidence || 0) / 100),
    0
  );

  let sentiment = "Neutral";
  if (score > 0.25) {
    sentiment = "Positive";
  } else if (score < -0.25) {
    sentiment = "Negative";
  }

  const confidence = Number(
    (items.reduce((sum, item) => sum + Number(item.confidence || 0), 0) / items.length).toFixed(2)
  );

  return { sentiment, confidence };
}

function detectLanguage(text) {
  const raw = String(text || "");
  const lower = ` ${raw.toLowerCase()} `;

  if (!raw.trim()) {
    return { code: "unknown", name: LANGUAGE_NAME_MAP.unknown, confidence: 0.0 };
  }

  if (/[\u0C80-\u0CFF]/.test(raw)) return { code: "kn", name: LANGUAGE_NAME_MAP.kn, confidence: 0.98 };
  if (/[\u0980-\u09FF]/.test(raw)) return { code: "bn", name: LANGUAGE_NAME_MAP.bn, confidence: 0.98 };
  if (/[\u0600-\u06FF]/.test(raw)) return { code: "ar", name: LANGUAGE_NAME_MAP.ar, confidence: 0.98 };
  if (/[\u0900-\u097F]/.test(raw)) return { code: "hi", name: LANGUAGE_NAME_MAP.hi, confidence: 0.98 };
  if (/[\uAC00-\uD7AF]/.test(raw)) return { code: "ko", name: LANGUAGE_NAME_MAP.ko, confidence: 0.98 };
  if (/[\u3040-\u30FF]/.test(raw)) return { code: "ja", name: LANGUAGE_NAME_MAP.ja, confidence: 0.98 };
  if (/[\u4E00-\u9FFF]/.test(raw)) return { code: "zh", name: LANGUAGE_NAME_MAP.zh, confidence: 0.96 };
  if (/[\u0400-\u04FF]/.test(raw)) return { code: "ru", name: LANGUAGE_NAME_MAP.ru, confidence: 0.96 };

  let bestCode = "en";
  let bestScore = 0;

  for (const [code, hints] of Object.entries(LATIN_LANGUAGE_HINTS)) {
    const score = hints.reduce((sum, hint) => sum + (lower.includes(hint) ? 1 : 0), 0);
    if (score > bestScore) {
      bestCode = code;
      bestScore = score;
    }
  }

  if (bestScore > 0) {
    return {
      code: bestCode,
      name: LANGUAGE_NAME_MAP[bestCode] || "Unknown",
      confidence: Math.min(0.92, 0.5 + bestScore * 0.08)
    };
  }

  return { code: "en", name: LANGUAGE_NAME_MAP.en, confidence: 0.58 };
}

async function translateToEnglish(text, sourceLanguage = "auto") {
  const source = String(sourceLanguage || "auto").toLowerCase();
  const payload = {
    q: String(text || ""),
    source,
    target: "en",
    format: "text"
  };

  const endpoints = [
    "https://libretranslate.de/translate",
    "https://translate.argosopentech.com/translate"
  ];

  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        continue;
      }

      const data = await response.json();
      const translated = String(data?.translatedText || "").trim();
      if (translated) {
        return { translatedText: translated, provider: endpoint };
      }
    } catch {
      // Try next endpoint.
    }
  }

  return null;
}

module.exports = async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const text = String(req.body?.text || "").trim();
    const mode = getRequestedMode(req.body?.mode);
    const autoDetectLanguage = req.body?.options?.autoDetectLanguage !== false;
    const translateBeforeAnalysis = Boolean(req.body?.options?.translateToEnglish);

    if (!text) {
      return res.status(400).json({ error: "Text is required" });
    }

    const language = autoDetectLanguage
      ? detectLanguage(text)
      : { code: "unknown", name: LANGUAGE_NAME_MAP.unknown, confidence: 0.0 };

    let textForAnalysis = text;
    let translationApplied = false;
    let translationWarning = "";
    let translationProvider = "";

    if (translateBeforeAnalysis) {
      const source = language.code === "unknown" ? "auto" : language.code;
      if (source !== "en") {
        const translated = await translateToEnglish(text, source);
        if (translated?.translatedText) {
          textForAnalysis = translated.translatedText;
          translationApplied = true;
          translationProvider = translated.provider;
        } else {
          translationWarning = "Translation service unavailable, analyzed original text.";
        }
      }
    }

    const languageMeta = {
      detected_code: language.code,
      detected_name: language.name,
      detected_confidence: Number(language.confidence.toFixed(2)),
      translation_requested: translateBeforeAnalysis,
      translation_applied: translationApplied,
      translation_provider: translationProvider || null,
      translated_text: translationApplied ? textForAnalysis : null
    };

    const token = cleanToken(
      process.env.HUGGINGFACE_API_TOKEN || process.env.HF_TOKEN || process.env.HUGGINGFACEHUB_API_TOKEN || ""
    );

    const configuredModel = cleanModelId(DEFAULT_MODEL);

    if (mode === "batch") {
      const lines = splitBatchInputs(textForAnalysis);
      if (lines.length === 0) {
        return res.status(400).json({ error: "Batch mode requires at least one non-empty line" });
      }

      const items = lines.map((line) => {
        const predicted = localFallbackSentiment(line);
        return {
          text: line,
          sentiment: predicted.sentiment,
          confidence: predicted.confidence
        };
      });

      const summary = aggregateBatch(items);

      return res.status(200).json({
        mode,
        sentiment: summary.sentiment,
        confidence: summary.confidence,
        items,
        language: languageMeta,
        model: "batch-local-fallback",
        source: "local-fallback",
        warning: translationWarning || "Batch mode currently uses local scoring per line."
      });
    }

    if (mode === "absa") {
      const absa = buildAbsaResult(textForAnalysis);
      return res.status(200).json({
        mode,
        sentiment: absa.overallSentiment,
        confidence: absa.overallConfidence,
        aspects: absa.aspects,
        language: languageMeta,
        model: "absa-heuristic",
        source: "local-absa",
        warning: translationWarning || undefined
      });
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
      let result = await requestInference(url, textForAnalysis, token, extraBody);
      response = result.response;
      data = result.data;
      lastErrorText = summarizeHfError(response, data);

      const permissionError = isAuthLikeError(response, data);

      // Some token types cannot call provider-routed inference. Retry once
      // anonymously for public models before failing.
      if (!response.ok && token && permissionError) {
        result = await requestInference(url, textForAnalysis, "", extraBody);
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
        const fallbackEmotion = emotionFromSentimentFallback(textForAnalysis);
        return res.status(200).json({
          mode,
          emotion: fallbackEmotion.emotion,
          confidence: fallbackEmotion.confidence,
          top_emotions: [{ label: fallbackEmotion.emotion, score: Number((fallbackEmotion.confidence / 100).toFixed(4)) }],
          language: languageMeta,
          model: "emotion-fallback",
          source: "local-fallback",
          reason: lastErrorText || "No successful response from inference endpoints",
          warning: translationWarning || undefined
        });
      }

      const fallback = localFallbackSentiment(textForAnalysis);
      return res.status(200).json({
        mode,
        sentiment: fallback.sentiment,
        confidence: fallback.confidence,
        language: languageMeta,
        model: "sentiment-js-fallback",
        source: "local-fallback",
        reason: lastErrorText || "No successful response from inference endpoints",
        warning: translationWarning || undefined
      });
    }

    if (mode === "emotion") {
      const predictions = flattenPredictions(data)
        .map((item) => ({ label: mapEmotionLabel(item.label), score: Number(item.score || 0) }))
        .sort((a, b) => b.score - a.score);

      const top = predictions[0] || { label: "Neutral", score: 0 };
      return res.status(200).json({
        mode,
        emotion: top.label,
        confidence: Number((top.score * 100).toFixed(2)),
        top_emotions: predictions.slice(0, 5),
        language: languageMeta,
        model: selectedModel,
        source: "huggingface",
        warning: translationWarning || undefined
      });
    }

    const prediction = bestPrediction(data);
    const normalized = toAppStyleSentiment(prediction);

    return res.status(200).json({
      mode,
      sentiment: normalized.sentiment,
      confidence: normalized.confidence,
      language: languageMeta,
      model: selectedModel,
      source: "huggingface",
      warning: translationWarning || undefined
    });
  } catch (error) {
    return res.status(500).json({
      error: "Unexpected server error",
      details: error.message
    });
  }
};
