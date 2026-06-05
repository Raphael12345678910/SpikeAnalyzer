import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("video");
const videoInput = document.getElementById("videoInput");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctx = overlayCanvas.getContext("2d");

const analyzeBtn = document.getElementById("analyzeBtn");
const resetBtn = document.getElementById("resetBtn");
const aiBtn = document.getElementById("aiBtn");
const aiCoach = document.getElementById("aiCoach");

const angleValue = document.getElementById("angleValue");
const extensionValue = document.getElementById("extensionValue");
const reachEfficiencyValue = document.getElementById("reachEfficiencyValue");
const contactReachValue = document.getElementById("contactReachValue");
const gainValue = document.getElementById("gainValue");
const netMarginValue = document.getElementById("netMarginValue");
const systemOutput = document.getElementById("systemOutput");

const hittingHandInput = document.getElementById("hittingHand");
const cameraAngleInput = document.getElementById("cameraAngle");
const repTypeInput = document.getElementById("repType");
const userNotesInput = document.getElementById("userNotes");
const standingReachInput = document.getElementById("standingReach");
const netHeightInput = document.getElementById("netHeight");

let poseLandmarker;
let isAnalyzing = false;
let frames = [];
let lastAnalyzed = null;
let currentVideoUrl = null;
let latestLandmarks = null;
let analysisRunId = 0;

const ANALYSIS_FPS = 24;
const MIN_VISIBILITY = 0.45;

const LM = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftElbow: 13,
  rightElbow: 14,
  leftWrist: 15,
  rightWrist: 16,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28
};

const poseConnections = [
  [0, 1], [1, 2], [2, 3], [3, 7],
  [0, 4], [4, 5], [5, 6], [6, 8],
  [9, 10],
  [11, 12],
  [11, 13], [13, 15],
  [12, 14], [14, 16],
  [11, 23], [12, 24],
  [23, 24],
  [23, 25], [25, 27], [27, 29], [29, 31],
  [24, 26], [26, 28], [28, 30], [30, 32],
  [27, 31], [28, 32]
];

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function round1(n) {
  return Math.round(n * 10) / 10;
}

function round2(n) {
  return Math.round(n * 100) / 100;
}

function pct(n) {
  return Math.round(n * 100);
}

function angle(a, b, c) {
  const ab = { x: a.x - b.x, y: a.y - b.y };
  const cb = { x: c.x - b.x, y: c.y - b.y };

  const dot = ab.x * cb.x + ab.y * cb.y;
  const mag1 = Math.sqrt(ab.x * ab.x + ab.y * ab.y);
  const mag2 = Math.sqrt(cb.x * cb.x + cb.y * cb.y);

  if (!mag1 || !mag2) return null;

  let cos = dot / (mag1 * mag2);
  cos = clamp(cos, -1, 1);

  return Math.acos(cos) * 180 / Math.PI;
}

function extensionScore(a) {
  if (!Number.isFinite(a)) return null;
  return clamp(Math.round(((a - 125) / 50) * 9 + 1), 1, 10);
}

function armHeight(s, w) {
  return s.y - w.y;
}

function averagePoint(points) {
  const valid = points.filter(Boolean);
  if (!valid.length) return null;

  return {
    x: valid.reduce((sum, p) => sum + p.x, 0) / valid.length,
    y: valid.reduce((sum, p) => sum + p.y, 0) / valid.length,
    z: valid.reduce((sum, p) => sum + (p.z || 0), 0) / valid.length,
    visibility: valid.reduce((sum, p) => sum + (p.visibility ?? 1), 0) / valid.length
  };
}

function minVisibility(points) {
  return Math.min(...points.filter(Boolean).map((p) => p.visibility ?? 1));
}

function normalize(value, min, max) {
  if (!Number.isFinite(value) || max === min) return 0;
  return clamp((value - min) / (max - min), 0, 1);
}

function formatSignedSeconds(seconds) {
  if (!Number.isFinite(seconds)) return "unknown";
  const sign = seconds > 0 ? "+" : "";
  return `${sign}${round2(seconds)}s`;
}

function timingLabel(deltaFromPeak) {
  if (!Number.isFinite(deltaFromPeak)) return "Unknown";
  if (deltaFromPeak < -0.12) return "Early";
  if (deltaFromPeak > 0.12) return "Late";
  return "On time";
}

function timingScore(deltaFromPeak) {
  if (!Number.isFinite(deltaFromPeak)) return null;
  return clamp(Math.round(10 - Math.max(0, Math.abs(deltaFromPeak) - 0.05) * 28), 1, 10);
}

function confidenceLabel(score) {
  if (score >= 0.78) return "High";
  if (score >= 0.58) return "Medium";
  return "Low";
}

function jointSet(lm, hand) {
  const shoulder = lm[hand ? LM.leftShoulder : LM.rightShoulder];
  const elbow = lm[hand ? LM.leftElbow : LM.rightElbow];
  const wrist = lm[hand ? LM.leftWrist : LM.rightWrist];
  const hip = lm[hand ? LM.leftHip : LM.rightHip];

  const shoulderCenter = averagePoint([lm[LM.leftShoulder], lm[LM.rightShoulder]]);
  const hipCenter = averagePoint([lm[LM.leftHip], lm[LM.rightHip]]);
  const bodyCenter = averagePoint([shoulderCenter, hipCenter]);
  const ankleCenter = averagePoint([lm[LM.leftAnkle], lm[LM.rightAnkle]]);
  const kneeCenter = averagePoint([lm[LM.leftKnee], lm[LM.rightKnee]]);

  return {
    shoulder,
    elbow,
    wrist,
    hip,
    shoulderCenter,
    hipCenter,
    bodyCenter,
    kneeCenter,
    ankleCenter
  };
}

function buildFrame(t, lm, hand) {
  const joints = jointSet(lm, hand);
  const { shoulder, elbow, wrist, hip, shoulderCenter, hipCenter, bodyCenter } = joints;

  if (!shoulder || !elbow || !wrist || !hip || !shoulderCenter || !hipCenter || !bodyCenter) {
    return null;
  }

  const visibility = minVisibility([shoulder, elbow, wrist, hip, shoulderCenter, hipCenter]);
  if (visibility < MIN_VISIBILITY) return null;

  const elbowAngle = angle(shoulder, elbow, wrist);
  const shoulderAngle = angle(hip, shoulder, wrist);
  const reachHeight = armHeight(shoulder, wrist);
  const wristHeight = 1 - wrist.y;
  const bodyHeight = 1 - bodyCenter.y;

  if (
    !Number.isFinite(elbowAngle) ||
    !Number.isFinite(shoulderAngle) ||
    !Number.isFinite(reachHeight) ||
    !Number.isFinite(wristHeight) ||
    !Number.isFinite(bodyHeight)
  ) {
    return null;
  }

  return {
    t,
    lm,
    joints,
    s: shoulder,
    e: elbow,
    w: wrist,
    ang: elbowAngle,
    shoulderAngle,
    h: reachHeight,
    wristHeight,
    bodyHeight,
    bodyCenterY: bodyCenter.y,
    visibility
  };
}

function smoothFrames(inputFrames) {
  return inputFrames.map((frame, index) => {
    const neighbors = inputFrames.slice(Math.max(0, index - 2), Math.min(inputFrames.length, index + 3));
    const avg = (key) => neighbors.reduce((sum, f) => sum + f[key], 0) / neighbors.length;

    return {
      ...frame,
      bodyCenterYSmooth: avg("bodyCenterY"),
      wristHeightSmooth: avg("wristHeight"),
      reachHeightSmooth: avg("h"),
      elbowAngleSmooth: avg("ang"),
      shoulderAngleSmooth: avg("shoulderAngle")
    };
  });
}

function addVelocities(inputFrames) {
  return inputFrames.map((frame, index) => {
    const prev = inputFrames[Math.max(0, index - 1)];
    const next = inputFrames[Math.min(inputFrames.length - 1, index + 1)];
    const dt = next.t - prev.t;

    if (!dt) {
      return {
        ...frame,
        bodyVelocityY: 0,
        wristVelocityY: 0
      };
    }

    return {
      ...frame,
      bodyVelocityY: (next.bodyCenterYSmooth - prev.bodyCenterYSmooth) / dt,
      wristVelocityY: ((1 - next.w.y) - (1 - prev.w.y)) / dt
    };
  });
}

function waitForSeek(targetTime, runId) {
  if (Math.abs(video.currentTime - targetTime) < 0.002) {
    return waitForFrameReady(runId);
  }

  return new Promise((resolve, reject) => {
    let settled = false;
    const timeout = window.setTimeout(() => {
      cleanup();
      if (runId !== analysisRunId) {
        resolve();
        return;
      }

      reject(new Error(`Timed out while seeking to ${round2(targetTime)}s.`));
    }, 2500);

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("seeked", onSeeked);
      video.removeEventListener("error", onError);
    };

    const onSeeked = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };

    const onError = () => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(new Error("Video seek failed."));
    };

    video.addEventListener("seeked", onSeeked);
    video.addEventListener("error", onError);
    video.currentTime = targetTime;
  });
}

function waitForFrameReady(runId) {
  if (video.readyState >= 2) {
    return new Promise((resolve) => window.requestAnimationFrame(resolve));
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      if (runId !== analysisRunId) {
        resolve();
        return;
      }

      reject(new Error("Timed out while waiting for the video frame."));
    }, 2500);

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadeddata", onReady);
      video.removeEventListener("canplay", onReady);
      video.removeEventListener("error", onError);
    };

    const onReady = () => {
      cleanup();
      window.requestAnimationFrame(resolve);
    };

    const onError = () => {
      cleanup();
      reject(new Error("Video frame failed to load."));
    };

    video.addEventListener("loadeddata", onReady);
    video.addEventListener("canplay", onReady);
    video.addEventListener("error", onError);
  });
}

function waitForMetadata() {
  if (Number.isFinite(video.duration) && video.duration > 0) {
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    const timeout = window.setTimeout(() => {
      cleanup();
      reject(new Error("Timed out while loading video metadata."));
    }, 3000);

    const cleanup = () => {
      window.clearTimeout(timeout);
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
      video.removeEventListener("error", onError);
    };

    const onLoadedMetadata = () => {
      cleanup();
      resolve();
    };

    const onError = () => {
      cleanup();
      reject(new Error("Video metadata failed to load."));
    };

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    video.addEventListener("error", onError);
  });
}

function getSampleTimes(duration) {
  const end = Math.max(0, duration - 0.05);
  const step = 1 / ANALYSIS_FPS;
  const times = [];

  for (let t = 0; t <= end; t += step) {
    times.push(round2(t));
  }

  if (!times.length || times[times.length - 1] < end) {
    times.push(round2(end));
  }

  return [...new Set(times)];
}

function analyzePoseFrames(usableFrames) {
  const processed = addVelocities(smoothFrames(usableFrames));
  const peakJump = [...processed].sort((a, b) => a.bodyCenterYSmooth - b.bodyCenterYSmooth)[0];

  const minWrist = Math.min(...processed.map((f) => f.w.y));
  const maxWrist = Math.max(...processed.map((f) => f.w.y));
  const minReach = Math.min(...processed.map((f) => f.reachHeightSmooth));
  const maxReach = Math.max(...processed.map((f) => f.reachHeightSmooth));
  const minBody = Math.min(...processed.map((f) => f.bodyCenterYSmooth));
  const maxBody = Math.max(...processed.map((f) => f.bodyCenterYSmooth));

  const candidates = processed.filter((f) => (
    f.reachHeightSmooth > 0 &&
    f.elbowAngleSmooth > 105 &&
    f.shoulderAngleSmooth > 85
  ));

  const pool = candidates.length >= 4 ? candidates : processed;

  for (const frame of pool) {
    const wristHigh = 1 - normalize(frame.w.y, minWrist, maxWrist);
    const reachHigh = normalize(frame.reachHeightSmooth, minReach, maxReach);
    const elbowExtended = normalize(frame.elbowAngleSmooth, 115, 175);
    const shoulderRaised = normalize(frame.shoulderAngleSmooth, 95, 170);
    const jumpHigh = 1 - normalize(frame.bodyCenterYSmooth, minBody, maxBody);

    frame.contactScore =
      wristHigh * 0.3 +
      reachHigh * 0.22 +
      elbowExtended * 0.24 +
      shoulderRaised * 0.14 +
      jumpHigh * 0.1;
  }

  const contact = [...pool].sort((a, b) => b.contactScore - a.contactScore)[0];
  const deltaFromPeak = contact.t - peakJump.t;
  const reachPercent = maxReach > 0 ? clamp(contact.reachHeightSmooth / maxReach, 0, 1) : null;
  const visibilityAvg = processed.reduce((sum, f) => sum + f.visibility, 0) / processed.length;
  const confidenceScore = clamp(
    visibilityAvg * 0.45 +
      (pool.length / processed.length) * 0.15 +
      contact.contactScore * 0.3 +
      (cameraAngleInput.value === "side" ? 0.1 : 0.04),
    0,
    1
  );

  const warnings = [];

  if (cameraAngleInput.value !== "side") {
    warnings.push("side-view camera angle is recommended for cleaner timing and arm-extension estimates");
  }

  if (visibilityAvg < 0.7) {
    warnings.push("some landmarks had low visibility, so the estimate may be noisy");
  }

  if (deltaFromPeak > 0.12) {
    warnings.push("likely contact frame is after jump peak, so the athlete may be contacting while descending");
  } else if (deltaFromPeak < -0.12) {
    warnings.push("likely contact frame is before jump peak, so the athlete may be contacting before full jump height");
  }

  if (contact.elbowAngleSmooth < 150) {
    warnings.push("hitting elbow is not fully extended near the likely contact frame");
  }

  if (contact.shoulderAngleSmooth < 130) {
    warnings.push("hitting arm is not reaching high enough above the shoulder near the likely contact frame");
  }

  warnings.push("no ball tracking yet, so contact timing is inferred from body position");

  return {
    processed,
    contact,
    peakJump,
    deltaFromPeak,
    reachPercent,
    timing: timingLabel(deltaFromPeak),
    timingScore: timingScore(deltaFromPeak),
    extensionScore: extensionScore(contact.elbowAngleSmooth),
    reachEfficiencyScore: reachPercent === null ? null : clamp(Math.round(reachPercent * 10), 1, 10),
    confidenceScore,
    confidenceLabel: confidenceLabel(confidenceScore),
    warnings
  };
}

async function runDeterministicAnalysis(runId) {
  await waitForMetadata();

  const sampleTimes = getSampleTimes(video.duration);
  const timestampBase = performance.now();

  for (let i = 0; i < sampleTimes.length; i += 1) {
    if (!isAnalyzing || runId !== analysisRunId) return;

    const t = sampleTimes[i];
    await waitForSeek(t, runId);

    if (!isAnalyzing || runId !== analysisRunId) return;

    const result = poseLandmarker.detectForVideo(video, timestampBase + t * 1000);

    if (result.landmarks && result.landmarks.length > 0) {
      const lm = result.landmarks[0];
      const hand = hittingHandInput.value === "left";
      const frame = buildFrame(t, lm, hand);

      if (frame) {
        frames.push(frame);
      }

      latestLandmarks = lm;
      drawFrame(lm);
    }

    if (i % 8 === 0 || i === sampleTimes.length - 1) {
      const progress = Math.round(((i + 1) / sampleTimes.length) * 100);
      systemOutput.textContent = `Analyzing exact frames... ${progress}%`;
      await new Promise((resolve) => window.requestAnimationFrame(resolve));
    }
  }

  finishAnalysis();
}

function primaryAdvice(metrics) {
  const { contact, deltaFromPeak, reachPercent } = metrics;

  if (deltaFromPeak > 0.12) {
    return {
      issue: "You are probably contacting after your jump peak.",
      cue: "Start your arm swing a little sooner so your hand is high as your body reaches its top point.",
      drill: "Try toss-to-self spike reps where the goal is to freeze contact at the top of the jump, then compare the contact frame to the peak-jump frame."
    };
  }

  if (deltaFromPeak < -0.12) {
    return {
      issue: "You are probably reaching for the ball before your full jump height.",
      cue: "Let the last two steps load the jump, then delay the arm strike until your torso has finished rising.",
      drill: "Use approach-jump catches: catch the ball with your hitting hand as high as possible, focusing on waiting until the top."
    };
  }

  if (contact.elbowAngleSmooth < 150) {
    return {
      issue: "Your hitting arm looks bent near the likely contact frame.",
      cue: "Reach through the ball with a taller elbow and finish contact with your hand above and slightly in front of your hitting shoulder.",
      drill: "Do wall snaps from a high reach position, starting with a straight arm and snapping through without letting the elbow collapse."
    };
  }

  if (reachPercent !== null && reachPercent < 0.9) {
    return {
      issue: "You are not using your best available reach at the likely contact frame.",
      cue: "Drive up first, then swing through a high contact point instead of letting the hand drop into the ball.",
      drill: "Mark your highest wrist point from the video and repeat approach jumps trying to match that height at contact."
    };
  }

  return {
    issue: "Your timing and extension look reasonably coordinated in this clip.",
    cue: "Keep chasing a high, fully extended contact while making the approach rhythm repeatable.",
    drill: "Alternate full-speed reps with one controlled rep where you pause the video and check that contact is near jump peak."
  };
}

function drawFrame(landmarks) {
  if (!landmarks) return;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const cw = overlayCanvas.width;
  const ch = overlayCanvas.height;

  if (!vw || !vh || !cw || !ch) return;

  const va = vw / vh;
  const ca = cw / ch;

  let w, h, ox, oy;

  if (va > ca) {
    w = cw;
    h = cw / va;
    ox = 0;
    oy = (ch - h) / 2;
  } else {
    h = ch;
    w = ch * va;
    oy = 0;
    ox = (cw - w) / 2;
  }

  const pts = landmarks.map((l) => ({
    x: ox + l.x * w,
    y: oy + l.y * h,
    z: l.z,
    visibility: l.visibility
  }));

  ctx.save();
  ctx.lineWidth = 4;
  ctx.strokeStyle = "#ffffff";
  ctx.fillStyle = "#ffffff";

  for (const [start, end] of poseConnections) {
    const a = pts[start];
    const b = pts[end];
    if (!a || !b) continue;

    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(b.x, b.y);
    ctx.stroke();
  }

  for (const point of pts) {
    if (!point) continue;
    ctx.beginPath();
    ctx.arc(point.x, point.y, 4, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.restore();
}

function finishAnalysis() {
  if (!isAnalyzing) return;

  isAnalyzing = false;
  video.pause();

  if (frames.length < 10) {
    systemOutput.textContent = "Not enough frames. Try a cleaner side-view clip.";
    return;
  }

  const usableFrames = frames.filter((f) => (
    Number.isFinite(f.ang) &&
    Number.isFinite(f.h) &&
    Number.isFinite(f.bodyCenterY) &&
    Number.isFinite(f.shoulderAngle)
  ));

  if (usableFrames.length < 10) {
    systemOutput.textContent = "Pose was detected, but not enough reliable arm frames were found. Try a clearer clip.";
    return;
  }

  const metrics = analyzePoseFrames(usableFrames);
  const { contact, peakJump, deltaFromPeak, reachPercent, warnings } = metrics;
  const advice = primaryAdvice(metrics);

  angleValue.textContent = round1(contact.elbowAngleSmooth) + "°";
  extensionValue.textContent = metrics.extensionScore + "/10";
  reachEfficiencyValue.textContent = metrics.reachEfficiencyScore + "/10";

  contactReachValue.textContent =
    `${metrics.timing} (${formatSignedSeconds(deltaFromPeak)} from peak)`;
  gainValue.textContent =
    reachPercent === null ? "Unknown" : `${pct(reachPercent)}% of best reach`;
  netMarginValue.textContent =
    `${metrics.confidenceLabel} (${pct(metrics.confidenceScore)}%)`;

  const warningLines = warnings.map((warning) => `- ${warning}`).join("\n");

  systemOutput.textContent =
    "Analysis complete\n" +
    "Likely contact frame: " + round2(contact.t) + "s\n" +
    "Peak jump frame: " + round2(peakJump.t) + "s\n" +
    "Timing: " + metrics.timing + " (" + formatSignedSeconds(deltaFromPeak) + " from peak jump)\n" +
    "Main takeaway: " + advice.issue + "\n" +
    "Cue: " + advice.cue + "\n\n" +
    "Warnings:\n" + warningLines;

  lastAnalyzed = {
    elbowAngle: round1(contact.elbowAngleSmooth),
    shoulderAngle: round1(contact.shoulderAngleSmooth),
    extensionScore: metrics.extensionScore,
    reachEfficiencyScore: metrics.reachEfficiencyScore,
    timingScore: metrics.timingScore,
    timing: metrics.timing,
    likelyContactTime: round2(contact.t),
    peakJumpTime: round2(peakJump.t),
    timeFromPeak: round2(deltaFromPeak),
    reachPercent: reachPercent === null ? null : pct(reachPercent),
    confidence: metrics.confidenceLabel,
    confidenceScore: pct(metrics.confidenceScore),
    primaryIssue: advice.issue,
    cue: advice.cue,
    drill: advice.drill,
    warnings
  };

  aiBtn.disabled = false;

  const handleSeek = () => {
    drawFrame(contact.lm);
    video.removeEventListener("seeked", handleSeek);
  };

  if (Math.abs(video.currentTime - contact.t) < 0.002) {
    drawFrame(contact.lm);
  } else {
    video.addEventListener("seeked", handleSeek);
    video.currentTime = contact.t;
  }
}

function resetUI() {
  angleValue.textContent = "--";
  extensionValue.textContent = "--";
  reachEfficiencyValue.textContent = "--";
  contactReachValue.textContent = "--";
  gainValue.textContent = "--";
  netMarginValue.textContent = "--";
  systemOutput.textContent = "No analysis yet.";
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";
  aiBtn.disabled = true;

  isAnalyzing = false;
  analysisRunId += 1;

  frames = [];
  lastAnalyzed = null;
  latestLandmarks = null;

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
}

function resizeCanvasToVideoBox() {
  const rect = video.getBoundingClientRect();
  overlayCanvas.width = Math.max(1, Math.floor(rect.width));
  overlayCanvas.height = Math.max(1, Math.floor(rect.height));
}

videoInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  if (currentVideoUrl) {
    URL.revokeObjectURL(currentVideoUrl);
  }

  currentVideoUrl = URL.createObjectURL(file);
  video.src = currentVideoUrl;
  video.load();

  resetUI();
  systemOutput.textContent = "Video loaded. Click “Analyze Video” when ready.";
});

resetBtn.addEventListener("click", () => {
  if (currentVideoUrl) {
    URL.revokeObjectURL(currentVideoUrl);
    currentVideoUrl = null;
  }

  video.removeAttribute("src");
  video.load();
  videoInput.value = "";
  resetUI();
});

analyzeBtn.addEventListener("click", async () => {
  if (!video.src) {
    systemOutput.textContent = "Upload a video first.";
    return;
  }

  if (!poseLandmarker) {
    systemOutput.textContent = "Pose model is still loading.";
    return;
  }

  frames = [];
  latestLandmarks = null;
  lastAnalyzed = null;
  aiBtn.disabled = true;
  aiCoach.textContent = "Run the analysis first, then click “Get AI Coaching.”";

  resizeCanvasToVideoBox();

  isAnalyzing = true;
  analysisRunId += 1;
  const runId = analysisRunId;

  systemOutput.textContent = "Analyzing exact frames...";

  try {
    video.pause();
    await runDeterministicAnalysis(runId);
  } catch (error) {
    console.error(error);
    if (runId !== analysisRunId) return;
    isAnalyzing = false;
    systemOutput.textContent = `Could not analyze the video: ${error.message}`;
  }
});

aiBtn.addEventListener("click", async () => {
  if (!lastAnalyzed) {
    aiCoach.textContent = "Run analysis first.";
    return;
  }

  aiCoach.textContent = "Loading...";

  try {
    const res = await fetch("/api/coach", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        hittingHand: hittingHandInput.value,
        cameraAngle: cameraAngleInput.value,
        repType: repTypeInput.value,
        userNotes: userNotesInput.value || "",
        standingReach: standingReachInput.value || "",
        netHeight: netHeightInput.value || "",
        elbowAngle: lastAnalyzed.elbowAngle,
        shoulderAngle: lastAnalyzed.shoulderAngle,
        extensionScore: lastAnalyzed.extensionScore,
        reachEfficiencyScore: lastAnalyzed.reachEfficiencyScore,
        timingScore: lastAnalyzed.timingScore,
        timing: lastAnalyzed.timing,
        likelyContactTime: lastAnalyzed.likelyContactTime,
        peakJumpTime: lastAnalyzed.peakJumpTime,
        timeFromPeak: lastAnalyzed.timeFromPeak,
        reachPercent: lastAnalyzed.reachPercent,
        confidence: lastAnalyzed.confidence,
        confidenceScore: lastAnalyzed.confidenceScore,
        primaryIssue: lastAnalyzed.primaryIssue,
        cue: lastAnalyzed.cue,
        drill: lastAnalyzed.drill,
        warnings: lastAnalyzed.warnings
      })
    });

    const rawText = await res.text();
    const contentType = res.headers.get("content-type") || "";
    let data;

    try {
      if (!contentType.includes("application/json")) {
        const isHtml = rawText.trim().toLowerCase().startsWith("<!doctype html");

        throw new Error(
          isHtml
            ? "The AI coach API returned an HTML page instead of JSON. Start the app with npm run dev or deploy it with a working /api/coach route."
            : `The AI coach API returned ${contentType || "an unknown content type"} instead of JSON.`
        );
      }

      data = JSON.parse(rawText);
    } catch (error) {
      throw new Error(
        error.message ||
        (rawText.toLowerCase().includes("page could not be found")
          ? "The /api/coach route was not found on Vercel. Check that the folder is named api, then commit, push, and redeploy."
          : `Backend returned non-JSON: ${rawText.slice(0, 80)}`)
      );
    }

    if (!res.ok) {
      throw new Error(data.error || "Coach backend failed.");
    }

    aiCoach.textContent = data.feedback || "No response.";
  } catch (err) {
    console.error(err);
    aiCoach.textContent = `AI failed: ${err.message}`;
  }
});

async function init() {
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
    },
    runningMode: "VIDEO",
    numPoses: 1
  });

  systemOutput.textContent = "Ready. Upload a clip, then click “Analyze Video.”";
}

window.addEventListener("resize", () => {
  if (video.src) {
    resizeCanvasToVideoBox();
    if (latestLandmarks) {
      drawFrame(latestLandmarks);
    }
  }
});

init();
