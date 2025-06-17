import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

const LANDMARKS = {
  RIGHT_EYE_OUTER: 33,
  LEFT_EYE_OUTER: 263,
  LEFT_EAR: 234,
  RIGHT_EAR: 454,
};

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let faceLandmarker;

function makeEar() {
  const earsImage = new Image();
  earsImage.src = "monkey-ear.png"; // Ensure this image has transparency
  return earsImage;
}

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.classList.remove("hidden");
  video.srcObject = stream;
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadFaceLandmarker() {
  const filesetResolver = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
  );
  // Provide the model path and options for the FaceLandmarker
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
      delegate: "GPU",
    },
    outputFaceBlendshapes: true,
    runningMode: "LIVE_STREAM",
    numFaces: 2,
  });
}

function getRollAngle(landmarks) {
  const leftEye = landmarks[LANDMARKS.RIGHT_EYE_OUTER]; // Approximate left eye corner
  const rightEye = landmarks[LANDMARKS.LEFT_EYE_OUTER]; // Approximate right eye corner

  const deltaX = rightEye.x - leftEye.x;
  const deltaY = rightEye.y - leftEye.y;

  const angleRadians = Math.atan2(deltaY, deltaX);
  return angleRadians;
}

function drawEars(landmarks) {
  const earsImage = makeEar(); // Create a new ear image each time to ensure it's loaded
  const leftEar = landmarks[LANDMARKS.LEFT_EAR]; // Approximate left ear position
  const rightEar = landmarks[LANDMARKS.RIGHT_EAR]; // Approximate right ear position

  const rollAngle = getRollAngle(landmarks);

  const earWidth = 100; // Adjust as needed
  const earHeight = 100; // Adjust as needed

  // Draw left ear (mirrored)
  ctx.save();
  ctx.translate(leftEar.x * canvas.width - 26, leftEar.y * canvas.height - 10);
  ctx.rotate(rollAngle);

  ctx.drawImage(earsImage, -earWidth / 2, -earHeight / 2, earWidth, earHeight);
  ctx.restore();

  // Draw right ear (normal)
  ctx.save();
  ctx.translate(
    rightEar.x * canvas.width + 26,
    rightEar.y * canvas.height - 10
  );
  ctx.rotate(rollAngle);
  ctx.scale(-1, 1); // Flip horizontally
  ctx.drawImage(earsImage, -earWidth / 2, -earHeight / 2, earWidth, earHeight);
  ctx.restore();
}

async function render() {
  canvas.classList.remove("hidden");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;

  const results = await faceLandmarker.detectForVideo(video, performance.now());

  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.faceLandmarks && results.faceLandmarks.length > 0) {
    drawEars(results.faceLandmarks[0]);
  }

  requestAnimationFrame(render);
}

async function main() {
  const button = document.getElementById("startButton");

  button.addEventListener("click", async () => {
    console.log("Starting face landmarker...");
    button.disabled = true;
    button.classList.add("hidden");

    await setupCamera();
    await loadFaceLandmarker();
    render();
  });
}

main();
