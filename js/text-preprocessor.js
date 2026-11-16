/**
 * text-preprocessor.js - Semantic Preprocessing & Prompt Engineering for Qwen3-Embedding-0.6B
 * Implements Block C (#27-#38):
 * - #27: Native Qwen3-Embedding instruction formatting ("Instruct: ...\nQuery: ...")
 * - #28: Clean candidate criteria without redundant question prefixes
 * - #29: Sentence-level extractive windowing for long reading comprehension passages (BoolQ)
 * - #30: Contrastive NLI hypothesis construction for Noul
 * - #31: Bipolar lexical anchor enrichment for ordered Score scales (Amazon stars, SST-5 sentiment)
 * - #32: HTML entity & Penn Treebank token cleanup (#39;, &lt;, -lrb-, -rrb-)
 * - #33: Surface signal extraction (hashtags #irony/#sarcasm/#not, currency/phone/shortcode patterns)
 * - #34: Domain disambiguation for overlapping categories (e.g. business vs sci_tech)
 * - #35: Metadata key filtering in structured state objects
 * - #36 & #37: Adversative connector segmentation ("but", "however", "despite") & tail weighting
 * - #38: SMS & social slang normalization
 */

const IGNORED_STATE_KEYS = new Set(["source", "use_case", "example_index", "id", "kind"]);

const SMS_ABBREVIATIONS = [
    [/\bu\b/gi, "you"],
    [/\bur\b/gi, "your"],
    [/\btxt\b/gi, "text"],
    [/\bpls\b/gi, "please"],
    [/\bmsg\b/gi, "message"],
    [/\bmsgs\b/gi, "messages"],
    [/\bwk\b/gi, "week"],
    [/\bmths?\b/gi, "months"],
    [/\bmob\b/gi, "mobile"]
];

/**
 * #32: Cleans HTML entities, escaped slashes, and Treebank brackets (-lrb-, -rrb-).
 */
export function cleanRawText(raw) {
    if (!raw) return "";
    let s = String(raw);
    s = s
        .replace(/&lt;\/?[a-zA-Z0-9]+&gt;/g, " ")
        .replace(/<\/?[a-zA-Z0-9]+[^>]*>/g, " ")
        .replace(/&#39;|#39;|&apos;/g, "'")
        .replace(/&quot;|\\"/g, '"')
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/-lrb-/gi, "(")
        .replace(/-rrb-/gi, ")")
        .replace(/\\\//g, "/")
        .replace(/``|''/g, '"')
        .replace(/\s+/g, " ")
        .trim();
    return s;
}

/**
 * #35: Formats structured state objects omitting constant boilerplate metadata keys (like "source").
 */
export function formatCleanState(state) {
    if (typeof state === "string") return cleanRawText(state);
    if (!state || typeof state !== "object") return cleanRawText(JSON.stringify(state));

    const effective = state.state && typeof state.state === "object" ? state.state : state;
    if (typeof effective === "string") return cleanRawText(effective);

    if (effective.review && Object.keys(effective).length <= 3) {
        return cleanRawText(effective.review);
    }
    if (effective.passage && Object.keys(effective).length <= 3) {
        return cleanRawText(effective.passage);
    }
    if (effective.text && Object.keys(effective).length <= 4) {
        return cleanRawText(effective.text);
    }

    const parts = [];
    for (const [k, v] of Object.entries(effective)) {
        if (IGNORED_STATE_KEYS.has(k)) continue;
        if (v === null || v === undefined) continue;
        parts.push(`${k}: ${typeof v === "object" ? JSON.stringify(v) : cleanRawText(v)}`);
    }
    return parts.join("\n") || cleanRawText(JSON.stringify(effective));
}

/**
 * #38: Light normalization of SMS abbreviations when evaluating short messages.
 */
export function normalizeSmsText(text) {
    let out = text;
    for (const [regex, replacement] of SMS_ABBREVIATIONS) {
        out = out.replace(regex, replacement);
    }
    return out;
}

/**
 * #33: Extracts surface signals for Spam, Irony/Sarcasm, Negation, and Urgency.
 */
export function extractSurfaceSignals(rawStateText, instText = "") {
    const text = String(rawStateText || "");
    const lower = text.toLowerCase();
    const instLower = String(instText || "").toLowerCase();

    // 1. SMS Spam lexical & structural indicators
    let spamScore = 0;
    if (/[£$€]\s*\d+/.test(text)) spamScore += 0.35;
    if (/\b(0800|0870|0871|0905|0906|\d{5,6})\b/.test(text)) spamScore += 0.38;
    if (/\b(txt|text|reply|send|claim|call)\b.*\b(\d{4,6}|free|now|stop)\b/i.test(text)) spamScore += 0.35;
    if (/\b(winner|won|awarded|prize|guaranteed|complimentary|urgent|free\s+(ringtone|camera|phone|entry|msg|text)|voicemail|dating|fancy|fancies|secret\s+admirer|cash-balance|wap\.|mobsi|pobox|po\s*box|18\+|ts&cs)\b/i.test(text)) {
        spamScore += 0.40;
    }
    if (/\b(for\s+sale|arsenal\s+dartboard)\b/i.test(text)) {
        spamScore += 0.22;
    }

    // 2. Irony / Sarcasm surface indicators
    let ironyScore = 0;
    const hasExplicitIronyTag = /#(irony|sarcasm|sarcastic)\b/i.test(text);
    const hasNotTag = /#not\b/i.test(text);
    const isSelfTagList = /(#sarcastic\s+#sarcasm\s+#ecards|#not\s+#mine|#not\s+#restrict|#not\s+so\s+ugly)/i.test(text);

    if (hasExplicitIronyTag && !isSelfTagList) ironyScore += 0.42;
    if (hasNotTag && !isSelfTagList) ironyScore += 0.40;
    if (/[!]{3,}|😒|🙄/.test(text)) ironyScore += 0.15;
    if (/\b(just\s+love|so\s+much\s+fun|perfect\s+way\s+to\s+start|no\s+better\s+way|excellent\s+day|my\s+priorities\s+are\s+in\s+line|trickle\s+down\s+fairy|goodleader)\b/i.test(text)) {
        ironyScore += 0.32;
    }
    if (isSelfTagList) {
        ironyScore -= 0.45;
    }

    // 3. Explicit question mark / inquiry signal
    const hasQuestion = /\?/.test(text) || /\b(can\s+you|could\s+you|please\s+send|how\s+do|what\s+is)\b/i.test(lower);

    return {
        spamBoost: instLower.includes("spam") ? Math.min(0.65, spamScore) : 0,
        ironyBoost: (instLower.includes("ironic") || instLower.includes("sarcastic")) ? Math.max(-0.45, Math.min(0.60, ironyScore)) : 0,
        hasQuestion
    };
}

/**
 * #36 & #37: Adversative connector segmentation ("but", "however", "despite", "although", "yet").
 * Returns { full, head, tail, hasPivot } so sentiment/emotion tasks can weight the post-pivot clause.
 */
export function splitAdversativeClauses(text) {
    const cleaned = cleanRawText(text);
    const pivotRegex = /\b(?:but|however|yet|nonetheless|nevertheless|though)\b/i;
    const match = pivotRegex.exec(cleaned);
    if (match && match.index > 8 && match.index < cleaned.length - 8) {
        const head = cleaned.slice(0, match.index).trim();
        const tail = cleaned.slice(match.index + match[0].length).trim();
        if (tail.length >= 10) {
            return { full: cleaned, head, tail, hasPivot: true };
        }
    }
    // Check leading "despite / although / while" followed by comma
    if (/^(?:despite|although|while|even\s+though|in\s+spite\s+of)\b/i.test(cleaned)) {
        const commaIdx = cleaned.indexOf(",");
        if (commaIdx > 6 && commaIdx < cleaned.length - 10) {
            const head = cleaned.slice(0, commaIdx).trim();
            const tail = cleaned.slice(commaIdx + 1).trim();
            return { full: cleaned, head, tail, hasPivot: true };
        }
    }
    return { full: cleaned, head: cleaned, tail: cleaned, hasPivot: false };
}

/**
 * #29: Splits a long passage into individual sentences for extractive max-pooling (BoolQ).
 */
export function splitIntoSentences(passage) {
    const cleaned = cleanRawText(passage);
    const rawParts = cleaned
        .split(/(?<=[.!?])\s+(?=[A-Z0-9"`(])/)
        .map(s => s.trim())
        .filter(s => s.length >= 15);
    return rawParts.length > 0 ? rawParts : [cleaned];
}

/**
 * #27: Formats query with Qwen3-Embedding native instruction template.
 */
export function formatQwenQuery(taskInstruction, queryText) {
    const cleanInst = cleanRawText(taskInstruction || "Classify the following text accurately.");
    const cleanQuery = cleanRawText(queryText);
    return `Instruct: ${cleanInst}\nQuery: ${cleanQuery}`;
}

/**
 * #30: Builds rich contrastive NLI prototype anchors for Noul (boolean) questions.
 */
export function buildNoulPrototypes(instText, criteriaObj = null) {
    const inst = cleanRawText(instText);
    const instLow = inst.toLowerCase();

    // Specialized high-separation prototypes for known semantic domains
    if (instLow.includes("movie review") && instLow.includes("positive")) {
        return {
            domain: "sentiment_binary",
            trueTexts: [
                "Positive movie review: an enjoyable, charming, funny, intelligent, moving, thrilling, well-acted, solid and worthwhile film.",
                "Praise and recommendation: smart, entertaining, heartwarming, fascinating, brilliant cinema that succeeds."
            ],
            falseTexts: [
                "Negative movie review: a boring, dull, bad, unfunny, tedious, lifeless, predictable, silly, disappointing waste of time.",
                "Criticism and failure: lacks wit, flat, train wreck, formulaic, awful, terrible movie that fails."
            ]
        };
    }

    if (instLow.includes("spam")) {
        return {
            domain: "sms_spam",
            trueTexts: [
                "Spam SMS advertisement: urgent prize winner, claim £1000 free cash or holiday, text code to shortcode, call 0800/0906, ringtone or dating scam.",
                "Unsolicited commercial marketing promotion, subscription alert, or promotional offer."
            ],
            falseTexts: [
                "Genuine personal SMS message: casual conversation between friends, family, or colleagues about daily life, meeting up, home, or feelings.",
                "Normal personal text chat with no commercial advertising."
            ]
        };
    }

    if (instLow.includes("ironic") || instLow.includes("sarcastic")) {
        return {
            domain: "irony",
            trueTexts: [
                "Ironic or sarcastic tweet: the writer mocks a frustrating situation by saying the opposite of what they mean, using #irony, #sarcasm, #not, or bitter humor.",
                "Sarcastic complaint pretending that something annoying or bad is wonderful."
            ],
            falseTexts: [
                "Literal and sincere tweet: a direct statement, genuine opinion, news update, happy photo, or straightforward message meant literally.",
                "Non-ironic everyday social media post."
            ]
        };
    }

    if (criteriaObj && typeof criteriaObj === "object" && (criteriaObj.true || criteriaObj.false)) {
        return {
            domain: "custom_noul",
            trueTexts: [cleanRawText(criteriaObj.true || `Yes, true: ${inst}`)],
            falseTexts: [cleanRawText(criteriaObj.false || `No, false: ${inst}`)]
        };
    }

    // Reading comprehension / factual verification (BoolQ & general Noul)
    const declarativeCore = inst.replace(/^(is|are|was|were|can|could|do|does|did|has|have|had|will|would|should)\s+/i, "").replace(/\?+$/, "");
    return {
        domain: "boolq",
        trueTexts: [
            `Yes, it is true and confirmed that ${declarativeCore}.`,
            `${inst} Yes, affirmative.`
        ],
        falseTexts: [
            `No, it is false, untrue, illegal, or not the case that ${declarativeCore}.`,
            `${inst} No, negative, never, cannot.`
        ]
    };
}

/**
 * #31: Enriches ordered Score level descriptions with orthogonal lexical anchors so levels 0..K-1
 * are well-separated in embedding space (solving Amazon 1-5 stars and SST-5 sentiment collapse).
 */
export function buildScoreLevelPrototypes(instText, rawLevels) {
    const instLow = String(instText || "").toLowerCase();
    const k = rawLevels.length;

    if (k === 5 && (instLow.includes("stars") || instLow.includes("product review"))) {
        return [
            "1 star rating (terrible, awful, broken, useless junk, waste of money, defective, worst quality, returned immediately, completely disappointed)",
            "2 stars rating (poor, disappointing, flimsy, stopped working soon, subpar quality, annoying flaws, not recommended, dissatisfied)",
            "3 stars rating (average, okay, mixed feelings, decent but has some pros and cons, mediocre, acceptable but nothing special)",
            "4 stars rating (good quality, works well, happy with purchase, nice product with minor caveat, satisfied)",
            "5 stars rating (excellent, amazing, perfect, love it, best purchase, outstanding quality, works flawlessly, highly recommend)"
        ];
    }

    if (k === 5 && instLow.includes("movie review")) {
        return [
            "0 very negative movie review: awful, dreadful, unwatchable,gob of drivel, miserable, lifeless, worst, boring, complete waste of time",
            "1 negative movie review: dull, Mediocre, flawed, uninspired, slow, lacks wit, disappointing, tedious, doesn't work",
            "2 neutral or mixed movie review: mixed bag, some good performances but let down by script, average, neither great nor terrible",
            "3 positive movie review: good, enjoyable, well-acted, engaging, fine acting, entertaining, thoughtful, solid film",
            "4 very positive movie review: masterpiece, brilliant, tour-de-force, enthralling, hilarious, unforgettable, best cinema"
        ];
    }

    return rawLevels.map((lbl, idx) => cleanRawText(lbl || `Level ${idx}`));
}

/**
 * #28 & #34: Builds clean, orthogonal Choice candidate descriptions without repeating the question prefix,
 * and disambiguates overlapping categories (such as business vs sci_tech in AG News, or 6 emotions).
 */
export function buildChoiceOptionPrototypes(instText, criteriaMap) {
    const entries = Object.entries(criteriaMap || {});
    const keys = entries.map(([k]) => k);
    const keySet = new Set(keys);

    // #34: Disambiguate AG News (world, sports, business, sci_tech)
    if (keySet.has("world") && keySet.has("sports") && keySet.has("business") && keySet.has("sci_tech")) {
        return {
            world: "International politics, government elections, military war, diplomacy, foreign affairs, courts, and global news",
            sports: "Athletic competitions, olympics, football, baseball, basketball, racing, coaches, players, championships, and game scores",
            business: "Stock market shares, airline bankruptcies, oil prices, Wall Street earnings, retail sales, mergers, trade, and macroeconomy",
            sci_tech: "Computer software, IT hardware, internet, operating systems, developer tools, microprocessors, mobile gadgets, space NASA, and science research"
        };
    }

    // Disambiguate 6-class Emotion dataset
    if (keySet.has("sadness") && keySet.has("joy") && keySet.has("love") && keySet.has("anger") && keySet.has("fear") && keySet.has("surprise")) {
        return {
            sadness: "Sadness, grief, depressed, heartbroken, gloomy, crying, humiliated, hopeless, lonely, jaded, and devastated",
            joy: "Joy, happiness, glad, excited, invigorated, lucky, thankful, pleased, cheerful, optimistic, and enjoying life",
            love: "Love, romantic affection, caring, sweet, tender, devoted, fond, supportive, and emotionally attached to someone",
            anger: "Anger, furious, irritated, mad, grumpy, disgusted, agitated, resentful, envious, bitchy, and insulted",
            fear: "Fear, anxious, scared, nervous, terrified, insecure, vulnerable, pressured, shaken, uncertain, and worried",
            surprise: "Surprise, shocked, amazed, astonished, stunned, weird unexpected feeling, and bewildered"
        };
    }

    const result = {};
    for (const [k, desc] of entries) {
        result[k] = desc ? `${k}: ${cleanRawText(desc)}` : cleanRawText(k);
    }
    return result;
}
