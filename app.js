import {
  PoseLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

const video = document.getElementById("video");
const videoInput = document.getElementById("videoInput");
const overlayCanvas = document.getElementById("overlayCanvas");
const ctx = overlayCanvas.getContext("2d");

const analyzeBtn = document.getElementById("analyzeBtn");
const markContactBtn = document.getElementById("markContactBtn");
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
let currentMetrics = null;

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

  const minReach = Math.min(...processed.map((f) => f.reachHeightSmooth));
  const maxReach = Math.max(...processed.map((f) => f.reachHeightSmooth));
  const visibilityAvg = processed.reduce((sum, f) => sum + f.visibility, 0) / processed.length;
  const confidenceScore = clamp(
    visibilityAvg * 0.45 +
      (processed.length >= 24 ? 0.2 : 0.1) +
      (cameraAngleInput.value === "side" ? 0.2 : 0.08) +
      normalize(maxReach - minReach, 0.05, 0.25) * 0.15,
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

  warnings.push("no ball tracking yet, so automatic analysis uses peak jump as the review frame");
  warnings.push("mark the real ball-contact frame manually before using contact timing feedback");

  return {
    processed,
    peakJump,
    maxReach,
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

function closestFrame(framesToSearch, time) {
  return [...framesToSearch].sort((a, b) => Math.abs(a.t - time) - Math.abs(b.t - time))[0] || null;
}

function primaryAdvice(analysis) {
  if (analysis.contactSource === "manual" && analysis.timeFromPeak > 0.12) {
    return {
      issue: "Your marked contact is after your jump peak.",
      cue: "Start the arm swing a little sooner so your hand is high as your body reaches its top point.",
      drill: "Use approach reps where you pause at the marked contact frame and check whether it is moving closer to the peak-jump frame."
    };
  }

  if (analysis.contactSource === "manual" && analysis.timeFromPeak < -0.12) {
    return {
      issue: "Your marked contact is before your full jump height.",
      cue: "Wait a fraction longer before the strike so your jump can finish rising.",
      drill: "Use approach-jump catches: catch the ball with your hitting hand at your highest point before going back to full swings."
    };
  }

  if (analysis.elbowAngle < 150) {
    return {
      issue: "Your hitting arm looks bent at the analysis frame.",
      cue: "Reach through the ball with a taller elbow and finish contact with your hand above and slightly in front of your hitting shoulder.",
      drill: "Do wall snaps from a high reach position, starting with a straight arm and snapping through without letting the elbow collapse."
    };
  }

  if (analysis.reachPercent !== null && analysis.reachPercent < 90) {
    return {
      issue: "You are not using your best available reach at the analysis frame.",
      cue: "Drive up first, then swing through a high contact point instead of letting the hand drop into the ball.",
      drill: "Mark your highest wrist point from the video and repeat approach jumps trying to match that height at contact."
    };
  }

  return {
    issue: "Your peak-jump position and arm extension look reasonably coordinated in this clip.",
    cue: "Keep chasing a high, fully extended contact while making the approach rhythm repeatable.",
    drill: "Alternate full-speed reps with one controlled rep where you pause the video at peak jump and then at true contact."
  };
}

function buildAnalysisForFrame(frame, metrics, contactSource) {
  const timeFromPeak = contactSource === "manual" ? frame.t - metrics.peakJump.t : null;
  const reachRatio = metrics.maxReach > 0 ? clamp(frame.reachHeightSmooth / metrics.maxReach, 0, 1) : null;
  const timing = contactSource === "manual" ? timingLabel(timeFromPeak) : "Not marked";
  const warnings = [...metrics.warnings];

  if (contactSource === "manual") {
    if (timeFromPeak > 0.12) {
      warnings.push("marked contact is after peak jump, so the athlete may be contacting while descending");
    } else if (timeFromPeak < -0.12) {
      warnings.push("marked contact is before peak jump, so the athlete may be contacting before full jump height");
    }
  }

  if (frame.elbowAngleSmooth < 150) {
    warnings.push("hitting elbow is not fully extended at the analysis frame");
  }

  if (frame.shoulderAngleSmooth < 130) {
    warnings.push("hitting arm is not reaching high enough above the shoulder at the analysis frame");
  }

  const analysis = {
    analysisFrameTime: round2(frame.t),
    analysisFrameLabel: contactSource === "manual" ? "Manual contact frame" : "Peak jump frame",
    contactSource,
    elbowAngle: round1(frame.elbowAngleSmooth),
    shoulderAngle: round1(frame.shoulderAngleSmooth),
    extensionScore: extensionScore(frame.elbowAngleSmooth),
    reachEfficiencyScore: reachRatio === null ? null : clamp(Math.round(reachRatio * 10), 1, 10),
    timingScore: contactSource === "manual" ? timingScore(timeFromPeak) : null,
    timing,
    markedContactTime: contactSource === "manual" ? round2(frame.t) : null,
    peakJumpTime: round2(metrics.peakJump.t),
    timeFromPeak: contactSource === "manual" ? round2(timeFromPeak) : null,
    reachPercent: reachRatio === null ? null : pct(reachRatio),
    confidence: metrics.confidenceLabel,
    confidenceScore: pct(metrics.confidenceScore),
    warnings
  };

  const advice = primaryAdvice(analysis);

  return {
    ...analysis,
    primaryIssue: advice.issue,
    cue: advice.cue,
    drill: advice.drill
  };
}

function updateResults(analysis) {
  angleValue.textContent = round1(analysis.elbowAngle) + "°";
  extensionValue.textContent = analysis.extensionScore + "/10";
  reachEfficiencyValue.textContent = analysis.reachEfficiencyScore + "/10";
  contactReachValue.textContent =
    analysis.contactSource === "manual"
      ? `${analysis.timing} (${formatSignedSeconds(analysis.timeFromPeak)} from peak)`
      : "Mark contact frame";
  gainValue.textContent =
    analysis.reachPercent === null ? "Unknown" : `${analysis.reachPercent}% of best reach`;
  netMarginValue.textContent =
    `${analysis.confidence} (${analysis.confidenceScore}%)`;

  const warningLines = analysis.warnings.map((warning) => `- ${warning}`).join("\n");
  const contactLine =
    analysis.contactSource === "manual"
      ? `Marked contact frame: ${analysis.markedContactTime}s\n`
      : "Contact frame: not marked\n";

  systemOutput.textContent =
    "Analysis complete\n" +
    `Analysis frame: ${analysis.analysisFrameLabel} at ${analysis.analysisFrameTime}s\n` +
    contactLine +
    `Peak jump frame: ${analysis.peakJumpTime}s\n` +
    `Timing: ${analysis.timing}\n` +
    `Main takeaway: ${analysis.primaryIssue}\n` +
    `Cue: ${analysis.cue}\n\n` +
    "Warnings:\n" + warningLines;
}

function seekToAnalysisFrame(frame) {
  const handleSeek = () => {
    drawFrame(frame.lm);
    video.removeEventListener("seeked", handleSeek);
  };

  if (Math.abs(video.currentTime - frame.t) < 0.002) {
    drawFrame(frame.lm);
  } else {
    video.addEventListener("seeked", handleSeek);
    video.currentTime = frame.t;
  }
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
  currentMetrics = metrics;
  lastAnalyzed = buildAnalysisForFrame(metrics.peakJump, metrics, "peak");
  updateResults(lastAnalyzed);

  aiBtn.disabled = false;
  markContactBtn.disabled = false;

  seekToAnalysisFrame(metrics.peakJump);
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
  markContactBtn.disabled = true;

  isAnalyzing = false;
  analysisRunId += 1;

  frames = [];
  lastAnalyzed = null;
  currentMetrics = null;
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

video.addEventListener("seeked", () => {
  if (isAnalyzing || !currentMetrics?.processed?.length) return;

  const frame = closestFrame(currentMetrics.processed, video.currentTime);
  if (frame) {
    drawFrame(frame.lm);
  }
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
  currentMetrics = null;
  aiBtn.disabled = true;
  markContactBtn.disabled = true;
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

markContactBtn.addEventListener("click", () => {
  if (!currentMetrics || !currentMetrics.processed.length) {
    systemOutput.textContent = "Run analysis first, then scrub to contact and mark it.";
    return;
  }

  const markedFrame = closestFrame(currentMetrics.processed, video.currentTime);

  if (!markedFrame) {
    systemOutput.textContent = "Could not match the current video time to an analyzed frame.";
    return;
  }

  lastAnalyzed = buildAnalysisForFrame(markedFrame, currentMetrics, "manual");
  updateResults(lastAnalyzed);
  drawFrame(markedFrame.lm);
});

function buildBrowserCoachFeedback(analysis, note) {
  const timingLine =
    analysis.contactSource === "manual"
      ? `Your marked contact is ${formatSignedSeconds(analysis.timeFromPeak)} from peak jump, which is ${analysis.timing.toLowerCase()}.`
      : "You have not marked the true ball-contact frame yet, so I am not making a contact-timing judgment.";

  return `Biggest issue:
${analysis.primaryIssue}

Why it matters:
${timingLine} The most useful feedback comes from comparing true contact with peak jump, then checking whether the hitting arm is long and high at that exact frame.

What to change:
- ${analysis.cue}
- Mark the real ball-contact frame before judging early, late, or on-time contact.
- Keep the hitting hand high and the elbow long through the ball.

One drill:
${analysis.drill}

Confidence / limitations:
${analysis.confidence} confidence (${analysis.confidenceScore}%). ${note} This is local fallback coaching, not an OpenAI response.`;
}

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
        analysisFrameLabel: lastAnalyzed.analysisFrameLabel,
        analysisFrameTime: lastAnalyzed.analysisFrameTime,
        contactSource: lastAnalyzed.contactSource,
        elbowAngle: lastAnalyzed.elbowAngle,
        shoulderAngle: lastAnalyzed.shoulderAngle,
        extensionScore: lastAnalyzed.extensionScore,
        reachEfficiencyScore: lastAnalyzed.reachEfficiencyScore,
        timingScore: lastAnalyzed.timingScore,
        timing: lastAnalyzed.timing,
        markedContactTime: lastAnalyzed.markedContactTime,
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

        data = {
          feedback: buildBrowserCoachFeedback(
            lastAnalyzed,
            isHtml
              ? "The /api/coach route returned the app HTML instead of JSON. Run with npm run dev, or deploy to a host that supports the api/coach.js serverless route."
              : `The /api/coach route returned ${contentType || "an unknown content type"} instead of JSON.`
          )
        };
      } else {
        data = JSON.parse(rawText);
      }
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
    aiCoach.textContent = buildBrowserCoachFeedback(
      lastAnalyzed,
      `The coach backend request failed: ${err.message}`
    );
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
