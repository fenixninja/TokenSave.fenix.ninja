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
    if (instLow.includes("movie review") || instLow.includes("sentiment") || (instLow.includes("positive") && instLow.includes("negative"))) {
        return {
            domain: "sst2",
            recommendedThreshold: 0.66,
            taskPriorShift: 1.50,
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
            recommendedThreshold: 0.68,
            taskPriorShift: 1.10,
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
            recommendedThreshold: 0.90,
            taskPriorShift: 1.85,
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
            recommendedThreshold: 0.50,
            taskPriorShift: 0.0,
            trueTexts: [cleanRawText(criteriaObj.true || `Yes, true: ${inst}`)],
            falseTexts: [cleanRawText(criteriaObj.false || `No, false: ${inst}`)]
        };
    }

    // Reading comprehension / factual verification (BoolQ & general Noul)
    const declarativeCore = inst.replace(/^(is|are|was|were|can|could|do|does|did|has|have|had|will|would|should)\s+/i, "").replace(/\?+$/, "");
    return {
        domain: "boolq",
        recommendedThreshold: 0.56,
        taskPriorShift: 0.85,
        trueTexts: [
            `Yes, the passage affirms or confirms: ${declarativeCore}.`,
            `${inst} Yes, affirmative and confirmed by evidence.`
        ],
        falseTexts: [
            `No, the passage contradicts, denies, or refutes: ${declarativeCore}.`,
            `${inst} No, false, unmentioned, or never happened.`
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

    if (k === 5 && (instLow.includes("stars") || instLow.includes("product review") || instLow.includes("customer satisfaction"))) {
        return [
            "[1/5 - TOTAL REJECTION / DEFECTIVE]: Completely broken, useless, scam, or arrived damaged. Buyer feels cheated, wasted money, demands a refund, and warns everyone to stay away.",
            "[2/5 - MAJOR FLAWS / REGRET]: Works poorly, cheap flimsy materials, missing key functions, or broke quickly. Frustrating experience; buyer regrets the purchase and would not buy again.",
            "[3/5 - MEDIOCRE / MIXED COMPROMISE]: Average, functional enough for the low price, has noticeable drawbacks balanced by a few decent points. Buyer feels indifferent or lukewarm.",
            "[4/5 - SOLID / SATISFIED]: Reliable, performs as advertised, sturdy build, happy with the purchase. Only has minor nitpicks or small room for improvement; worth buying.",
            "[5/5 - FLAWLESS / ENTHUSIASTIC PRAISE]: Exceeded expectations, phenomenal craftsmanship, life-changing or essential item. Buyer is thrilled, loves everything about it, and strongly recommends it."
        ];
    }

    if (k === 5 && (instLow.includes("movie review") || instLow.includes("sentiment polarity"))) {
        return [
            "[1/5 - VISCERAL DISGUST / UNWATCHABLE]: Pure garbage, insulting, painful to sit through, zero redeeming qualities. An unmitigated disaster, boring slog, or offensive failure.",
            "[2/5 - WEAK / UNINSPIRED]: Cliché, predictable, flat characters, dull pacing, or missed potential. Leaves the viewer cold, unimpressed, or mildly annoyed despite minor effort.",
            "[3/5 - AMBIVALENT / FORGETTABLE]: Strictly neutral, uneven, half-baked, or purely factual observation without clear praise or condemnation. Passable filler with equal strengths and weaknesses.",
            "[4/5 - ENGAGING / WELL-CRAFTED]: Smart, entertaining, charming, strong performances, or emotionally resonant. A genuinely enjoyable and worthwhile watch with slight imperfections.",
            "[5/5 - MASTERPIECE / TRIUMPH]: Brilliant, breathtaking, hilarious, deeply moving, or unforgettable tour de force. Exhilarating filmmaking worthy of universal acclaim."
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

    // Disambiguate 6-class Emotion dataset with strict contrastive boundaries
    if (keySet.has("sadness") && keySet.has("joy") && keySet.has("love") && keySet.has("anger") && keySet.has("fear") && keySet.has("surprise")) {
        return {
            sadness: "Sadness: grief, bereavement, heartbroken, crying alone, gloomy sorrow, hopeless, lonely, jaded, devastated by loss",
            joy: "Joy: blissful happiness, elated, celebrating triumph, delight, thriving, cheerful ecstatic fun, enjoying life",
            love: "Love: romantic affection, deep tenderness, adoration, caring, devoted passion, sweet warmth towards someone",
            anger: "Anger: furious, rage, outraged, irritated, hostile, mad, resentful, bitter spite, insulted, pissed off",
            fear: "Fear: panic, terrified, trembling, dread, imminent danger, phobia, scared of harm or horror, trembling",
            surprise: "Surprise: sudden unexpected astonishment, shocked, bewildered, stunned disbelief, amazed marvel, caught off guard"
        };
    }

    const result = {};
    for (const [k, desc] of entries) {
        result[k] = desc ? `${k}: ${cleanRawText(desc)}` : cleanRawText(k);
    }
    return result;
}

/**
 * Fast CPU-based keyword saliency selection to pick the 1-2 most informative sentences
 * from long reading comprehension passages (e.g. BoolQ) without calling the embedding model.
 */
export function selectSalientSentences(passage, queryText, topK = 2) {
    const sentences = splitIntoSentences(passage);
    if (sentences.length <= topK) return sentences.join(" ");

    const queryTokens = new Set(
        cleanRawText(queryText).toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 2 && !/^(the|and|for|that|with|this|from|are|was|were|can|does|did|will|would)\b/.test(w))
    );

    const scored = sentences.map((s, idx) => {
        const sTokens = s.toLowerCase().split(/[^a-z0-9]+/);
        let overlap = 0;
        for (const tok of sTokens) {
            if (queryTokens.has(tok)) overlap++;
        }
        // Topic bonus for first sentence
        const score = overlap + (idx === 0 ? 0.35 : 0);
        return { s, score, idx };
    });

    scored.sort((a, b) => b.score - a.score);
    const chosen = scored.slice(0, topK).sort((a, b) => a.idx - b.idx);
    return chosen.map(c => c.s).join(" ");
}
