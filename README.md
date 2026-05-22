# SpikeCheck

![JavaScript](https://img.shields.io/badge/JavaScript-ES%20Modules-f7df1e?style=for-the-badge&logo=javascript&logoColor=000)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Tracking-4285f4?style=for-the-badge&logo=google&logoColor=fff)
![OpenAI](https://img.shields.io/badge/OpenAI-AI%20Coaching-111827?style=for-the-badge&logo=openai&logoColor=fff)
![Vercel](https://img.shields.io/badge/Vercel-Serverless%20API-000?style=for-the-badge&logo=vercel&logoColor=fff)

SpikeCheck is a volleyball training web app that analyzes a player's hitting arm during a spike attempt. The app lets an athlete upload a side-view video, detects body landmarks with MediaPipe, estimates the likely contact frame, calculates arm-extension metrics, and sends the results to an AI coaching endpoint for clear, practical feedback.

This project was built to explore how computer vision and generative AI can make sports feedback more accessible. Instead of requiring expensive motion-capture equipment or a private coach for every rep, SpikeCheck turns a normal phone video into a focused technical breakdown.

## Project Goals

- Make volleyball form analysis easier for student athletes using only short video clips.
- Use pose estimation to turn movement into measurable feedback.
- Pair objective metrics with human-readable coaching advice.
- Design the app honestly by showing confidence limits instead of pretending imperfect estimates are exact.
- Build a full-stack prototype that combines frontend video processing, serverless API design, and AI-assisted feedback.

## Features

### Video Upload and Preview

Players can upload a local volleyball clip directly in the browser. The app displays the video in a responsive preview area and overlays a pose skeleton while analysis is running.

### Pose-Based Arm Tracking

SpikeCheck uses MediaPipe's Pose Landmarker model to identify the shoulder, elbow, and wrist of the selected hitting arm. It samples video frames during playback and stores the arm position over time.

### Contact Frame Estimate

Because the app does not currently track the ball, it estimates the likely contact moment using pose-based signals:

- high wrist position relative to the shoulder
- strong arm extension
- proximity to the peak reach frame

The app makes this limitation explicit in the interface so the user understands that the contact frame is an estimate.

### Performance Metrics

SpikeCheck currently reports:

| Metric | What It Means |
| --- | --- |
| Elbow angle at likely contact | Approximate angle between shoulder, elbow, and wrist near the estimated contact frame |
| Arm extension score | A 1-10 score based on how extended the hitting arm is |
| Reach efficiency score | A 1-10 estimate of how close the contact frame is to the player's peak reach |
| System notes | Plain-language analysis status, warnings, and frame timing |

Some metrics, such as contact reach, gain above standing reach, and net margin, are intentionally disabled for now because the current prototype does not yet have reliable camera calibration or ball tracking.

### AI Coaching

After analysis, the app sends the measured results to a serverless API route. The API uses OpenAI to generate a coaching response with:

- the biggest technical issue
- why it matters
- three actionable cues
- one drill
- a confidence and limitations section

The coaching prompt is designed to be practical and honest. It tells the model not to invent precision where the app's measurements are limited.

## Tech Stack

| Area | Technology |
| --- | --- |
| Frontend | HTML, CSS, JavaScript ES modules |
| Pose estimation | MediaPipe Tasks Vision Pose Landmarker |
| Video rendering | HTML5 video and canvas overlay |
| Backend | Vercel-style serverless function |
| AI coaching | OpenAI Responses API |
| Package management | npm |

## How It Works

1. The user uploads a video clip.
2. The browser loads the video and prepares a canvas overlay.
3. MediaPipe detects body landmarks frame by frame while the video plays.
4. The app isolates the selected hitting arm based on the user's handedness.
5. For each reliable frame, it calculates:
   - shoulder-elbow-wrist angle
   - wrist height relative to the shoulder
   - timestamp
6. The app identifies a likely contact frame by combining reach height and extension.
7. Results are shown in the UI.
8. If requested, the results are sent to `/api/coach`.
9. The API calls OpenAI and returns a structured coaching report.

## Project Structure

```text
Spike-Analyzer/
├── api/
│   └── coach.js        # Serverless API endpoint for AI-generated coaching
├── app.js              # Frontend video analysis, pose tracking, and UI logic
├── index.html          # Main user interface and styling
├── package.json        # Project metadata and OpenAI dependency
├── package-lock.json   # Locked npm dependency versions
└── README.md           # Project documentation
```

## Getting Started

### Prerequisites

- Node.js 18 or newer
- npm
- An OpenAI API key for AI coaching
- A browser with modern JavaScript module support

### Installation

Clone the repository and install dependencies:

```bash
git clone <your-repository-url>
cd Spike-Analyzer
npm install
```

### Environment Variables

Create a local environment file or configure these variables in your deployment platform:

```bash
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4.1-mini
```

`OPENAI_MODEL` is optional. If it is not provided, the API defaults to `gpt-4.1-mini`.

### Running Locally

Run the included local server so the static frontend and `/api/coach` endpoint are served together:

```bash
npm run dev
```

Then open:

```text
http://localhost:3000
```

You can also use Vercel CLI:

```bash
npx vercel dev
```

Then open the local URL shown in the terminal.

For frontend-only testing, you can serve the static files:

```bash
python3 -m http.server 8000
```

Then visit:

```text
http://localhost:8000
```

In frontend-only mode, video analysis can still run, but AI coaching will not work unless `/api/coach` is available.

## Using the App

1. Record a volleyball spike from the side.
2. Make sure the full body is visible.
3. Upload the clip in SpikeCheck.
4. Select hitting hand, camera angle, and rep type.
5. Click **Analyze Video**.
6. Review the measured results and system notes.
7. Click **Get AI Coaching** for a written technical breakdown.

## Best Video Conditions

For better results, use a clip with:

- a stable camera
- side-view framing
- good lighting
- the full body visible
- minimal motion blur
- one athlete clearly in frame

Pose estimation is less reliable when the athlete is partially blocked, too far from the camera, filmed from the front, or moving through heavy blur.

## Current Limitations

SpikeCheck is a prototype, so it is designed to be transparent about what it can and cannot measure.

- It does not track the volleyball yet.
- The contact frame is estimated from body position, not ball contact.
- It does not calibrate real-world distance from pixels.
- Contact reach, standing-reach gain, and net-margin estimates are disabled until they can be made reliable.
- It works best on side-view clips with one athlete.
- Pose detection accuracy depends on lighting, camera angle, and body visibility.

These limitations are intentionally surfaced in the UI and included in the AI coaching prompt.

## Design Decisions

### Honest Measurement Over False Precision

Sports analytics tools can become misleading when they report exact numbers from uncertain inputs. SpikeCheck avoids this by disabling metrics that are not trustworthy yet and by warning the AI coach when the data is estimated.

### Browser-First Analysis

Pose analysis runs in the browser, which keeps the uploaded video local during the measurement step. The backend only receives the summarized metrics needed for coaching feedback.

### Lightweight Full-Stack Architecture

The project uses plain HTML, CSS, and JavaScript instead of a heavy frontend framework. This keeps the prototype readable, easy to deploy, and focused on the core computer-vision workflow.

## Future Improvements

- Add ball tracking to identify true contact frames.
- Calibrate video scale using known measurements such as net height or court markings.
- Re-enable contact reach and net-margin metrics after calibration.
- Add side-by-side comparison between multiple reps.
- Store session history for progress tracking.
- Add exportable reports for athletes and coaches.
- Improve mobile layout for recording and analysis on the same device.
- Add confidence scoring for each detected pose frame.

## What I Learned

This project combines several areas of software engineering:

- computer vision with browser-based pose estimation
- geometric analysis using body landmarks
- asynchronous video processing
- canvas overlays and responsive UI design
- serverless API development
- prompt design for practical AI feedback
- responsible handling of uncertain model outputs

The most important lesson was that useful AI tools need more than impressive output. They need careful boundaries, clear assumptions, and interfaces that explain uncertainty to the user.

## Responsible Use

SpikeCheck is an educational prototype. It should be used as a training aid, not as a medical, recruiting, or official performance evaluation tool. Athletes should combine the feedback with coach input, safe training habits, and their own judgment.

## Author

Built by Raphael Ferrand as a student software project exploring the intersection of sports, computer vision, and AI-assisted coaching.
