import OpenAI from "openai";

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const analysis = normalizeAnalysis(req.body || {});

    if (!process.env.OPENAI_API_KEY) {
      return res.status(200).json({
        feedback: buildLocalCoaching(analysis, "OpenAI is not configured, so this is the local coaching fallback.")
      });
    }

    const client = new OpenAI({
      apiKey: process.env.OPENAI_API_KEY
    });

    const prompt = `
You are a clear, practical volleyball coach.

Use the provided measurements as evidence, but do not pretend they are perfect.
Do not invent exact ball-contact timing because the app does not track the ball yet.
If a metric seems weak or limited, say so directly.

Return exactly this format:

Biggest issue:
<1 short paragraph>

Why it matters:
<1 short paragraph>

What to change:
- <cue 1>
- <cue 2>
- <cue 3>

One drill:
<1 short paragraph>

Confidence / limitations:
<1 short paragraph>

Player analysis data:

- Hitting hand: ${analysis.hittingHand}
- Camera angle: ${analysis.cameraAngle}
- Rep type: ${analysis.repType}
- User notes: ${analysis.userNotes || "none"}

- Likely contact time: ${analysis.likelyContactTime}s
- Peak jump time: ${analysis.peakJumpTime}s
- Contact timing relative to peak jump: ${analysis.timeFromPeak}s
- Timing category: ${analysis.timing}
- Timing score: ${analysis.timingScore}/10

- Elbow angle near likely contact: ${analysis.elbowAngle} degrees
- Shoulder reach angle near likely contact: ${analysis.shoulderAngle} degrees
- Extension score: ${analysis.extensionScore}/10
- Reach efficiency score: ${analysis.reachEfficiencyScore}/10
- Arm reach at likely contact: ${analysis.reachPercent}% of best observed reach

- Initial analyzer takeaway: ${analysis.primaryIssue}
- Initial analyzer cue: ${analysis.cue}
- Suggested drill from analyzer: ${analysis.drill}
- Confidence: ${analysis.confidence} (${analysis.confidenceScore}%)

- Warnings: ${
      analysis.warnings.length > 0
        ? analysis.warnings.join(", ")
        : "none"
    }
`;

    const response = await client.responses.create({
      model: process.env.OPENAI_MODEL || "gpt-4.1-mini",
      input: prompt
    });

    return res.status(200).json({
      feedback: response.output_text || "No coaching feedback returned."
    });
  } catch (error) {
    console.error("Coach API error:", error);

    return res.status(200).json({
      feedback: buildLocalCoaching(
        normalizeAnalysis(req.body || {}),
        `OpenAI request failed, so this is the local coaching fallback. ${error?.message || ""}`.trim()
      )
    });
  }
}

function normalizeAnalysis(body) {
  return {
    hittingHand: clean(body.hittingHand, "unknown"),
    cameraAngle: clean(body.cameraAngle, "unknown"),
    repType: clean(body.repType, "unknown"),
    userNotes: clean(body.userNotes, ""),
    standingReach: clean(body.standingReach, ""),
    netHeight: clean(body.netHeight, ""),
    elbowAngle: numberOrUnknown(body.elbowAngle),
    shoulderAngle: numberOrUnknown(body.shoulderAngle),
    extensionScore: numberOrUnknown(body.extensionScore),
    reachEfficiencyScore: numberOrUnknown(body.reachEfficiencyScore),
    timingScore: numberOrUnknown(body.timingScore),
    timing: clean(body.timing, "unknown"),
    likelyContactTime: numberOrUnknown(body.likelyContactTime),
    peakJumpTime: numberOrUnknown(body.peakJumpTime),
    timeFromPeak: numberOrUnknown(body.timeFromPeak),
    reachPercent: numberOrUnknown(body.reachPercent),
    confidence: clean(body.confidence, "unknown"),
    confidenceScore: numberOrUnknown(body.confidenceScore),
    primaryIssue: clean(body.primaryIssue, "The likely contact frame needs review."),
    cue: clean(body.cue, "Reach high and keep the timing close to the top of the jump."),
    drill: clean(body.drill, "Use controlled approach reps and pause the video at the likely contact frame."),
    warnings: Array.isArray(body.warnings) ? body.warnings.filter(Boolean).map(String) : []
  };
}

function clean(value, fallback) {
  if (value === null || value === undefined) return fallback;
  const text = String(value).trim();
  return text || fallback;
}

function numberOrUnknown(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : "unknown";
}

function buildLocalCoaching(analysis, note) {
  const issue = chooseIssue(analysis);
  const cues = chooseCues(analysis, issue.kind);
  const drill = analysis.drill || chooseDrill(issue.kind);

  return `Biggest issue:
${issue.text}

Why it matters:
Volleyball contact is best when your hand is high, your arm is long, and the hit happens close to the top of the jump. If contact is early, late, or bent, you lose reach and power even when the approach looked athletic.

What to change:
- ${cues[0]}
- ${cues[1]}
- ${cues[2]}

One drill:
${drill}

Confidence / limitations:
${analysis.confidence} confidence (${analysis.confidenceScore}%). ${note} The app still does not track the ball, so treat the contact frame as a body-position estimate. Warnings: ${analysis.warnings.join("; ") || "none"}.`;
}

function chooseIssue(analysis) {
  if (analysis.timing === "Late" || Number(analysis.timeFromPeak) > 0.12) {
    return {
      kind: "late",
      text: `Your likely contact is ${analysis.timeFromPeak}s after peak jump, so you may be hitting while starting to descend.`
    };
  }

  if (analysis.timing === "Early" || Number(analysis.timeFromPeak) < -0.12) {
    return {
      kind: "early",
      text: `Your likely contact is ${analysis.timeFromPeak}s before peak jump, so you may be reaching for the ball before your full height.`
    };
  }

  if (Number(analysis.extensionScore) < 7 || Number(analysis.elbowAngle) < 150) {
    return {
      kind: "bent",
      text: `Your elbow angle near likely contact is ${analysis.elbowAngle} degrees, which suggests your hitting arm is not fully extended.`
    };
  }

  if (Number(analysis.reachEfficiencyScore) < 8) {
    return {
      kind: "reach",
      text: `Your contact reach is about ${analysis.reachPercent}% of your best observed reach in this clip, so you are leaving height unused.`
    };
  }

  return {
    kind: "solid",
    text: "Your timing and extension look reasonably coordinated in this clip; the next step is making that high-contact pattern repeatable."
  };
}

function chooseCues(analysis, kind) {
  const base = {
    late: [
      "Start the hitting-arm swing a little sooner after takeoff.",
      "Aim to meet the ball as your torso reaches its highest point, not after it starts dropping.",
      "Keep the hand high through contact instead of letting the ball pull the arm downward."
    ],
    early: [
      "Wait a fraction longer before the arm strike so your jump can finish rising.",
      "Use the last two steps to load upward, then swing through the top.",
      "Keep your eyes on the ball and avoid reaching forward before your body is fully up."
    ],
    bent: [
      "Reach through the ball with a taller elbow.",
      "Contact above and slightly in front of the hitting shoulder.",
      "Think long arm first, wrist snap second."
    ],
    reach: [
      "Drive upward before swinging forward.",
      "Keep the hitting shoulder tall as the hand approaches contact.",
      "Try to match your best wrist height at the moment you strike."
    ],
    solid: [
      "Keep the contact point high and slightly in front.",
      "Make the approach rhythm repeatable so the same timing shows up every rep.",
      "Use video pauses to confirm contact stays near peak jump."
    ]
  };

  if (analysis.cue && analysis.cue !== "unknown") {
    return [analysis.cue, ...base[kind]].slice(0, 3);
  }

  return base[kind];
}

function chooseDrill(kind) {
  if (kind === "late") {
    return "Run approach-jump reps with an early arm-swing cue, then pause the video and check whether contact moved closer to peak jump.";
  }

  if (kind === "early") {
    return "Use approach-jump catches: catch the ball with your hitting hand at your highest point before going back to full swings.";
  }

  if (kind === "bent") {
    return "Do high wall snaps from a fully reached position, keeping the elbow tall before the wrist snaps through.";
  }

  return "Alternate one controlled rep and one full-speed rep, checking that your wrist height at contact stays close to your best reach.";
}
