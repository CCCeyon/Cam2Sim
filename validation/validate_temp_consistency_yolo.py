from datetime import datetime
import os
import tempfile
import cv2
from ultralytics import YOLO

# torchvision bits for resizing/cropping
from torchvision import transforms
from PIL import Image

# 512x512 transform (shorter side -> 512, then center crop to 512x512)
IMAGE_TRANSFORMS_512 = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
    ]
)

def _preprocess_video_to_512x512(src_path: str, transforms_512=IMAGE_TRANSFORMS_512) -> str:
    """
    Read a video, resize+center-crop every frame to 512x512, and write a temp MP4.
    Returns path to the temp preprocessed video.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Use mp4v for broad compatibility; change to 'avc1' if your ffmpeg build supports it
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    tmpfile = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_path = tmpfile.name
    tmpfile.close()

    out = cv2.VideoWriter(tmp_path, fourcc, fps, (512, 512), isColor=True)
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for the preprocessed video.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            # BGR -> RGB -> PIL
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)
            # Apply 512x512 transform
            pil_512 = transforms_512(pil)
            # Back to BGR for cv2 writer
            frame_out = cv2.cvtColor(
                np.array(pil_512, dtype="uint8"), cv2.COLOR_RGB2BGR
            )
            out.write(frame_out)
    finally:
        cap.release()
        out.release()

    return tmp_path


# numpy import needed for converting PIL->np for cv2
import numpy as np


def track_vehicles(
    video_path,
    outdir="runs/track",
    name=None,
    model_path="yolov8s.pt",
    device=None,
    conf=0.25,
    iou=0.5,
    imgsz=960,          # inference size; output video size is enforced by preprocessing
    half=False,
    save_frames=False,
    classes=[2, 3, 5, 7],   # vehicles in COCO
    tracker="strongsort.yaml",
    persist=False,
    force_512_video=True,   # NEW: make the source frames 512x512 so saved video is 512x512
    keep_preprocessed=False # keep the intermediate 512x512 video for inspection
):
    """
    Run vehicle detection + tracking (YOLO + StrongSORT) and save an annotated 512x512 video if requested.

    If force_512_video=True (default), the input is first rewritten so every frame is 512x512
    using torchvision transforms (Resize->CenterCrop). The saved annotated MP4 and per-frame
    labels then correspond to 512x512 frames.
    """
    run_name = name or datetime.now().strftime("vehtrack_%Y%m%d_%H%M%S")
    model = YOLO(model_path)

    # Optionally preprocess to fixed-size frames so the *saved* video is 512x512
    preprocessed_path = None
    source = video_path
    if force_512_video:
        preprocessed_path = _preprocess_video_to_512x512(video_path)
        source = preprocessed_path

    _ = model.track(
        source=source,
        tracker=tracker,
        conf=conf,
        iou=iou,
        imgsz=imgsz,       # controls inference resolution (not the saved video size)
        device=device,
        half=half,
        classes=classes,
        stream=False,
        verbose=True,
        save=True,          # annotated video
        save_txt=True,      # per-frame labels
        save_frames=save_frames,
        project=outdir,
        name=run_name,
        exist_ok=True,
        persist=persist,
    )

    # Optionally clean up the temporary video
    if preprocessed_path and not keep_preprocessed:
        try:
            os.remove(preprocessed_path)
        except OSError:
            pass

    run_dir = os.path.join(outdir, run_name)
    print("\n=== Done! ===")
    print(f"Output run directory: {run_dir}")
    print(f"  - Annotated video(s): {run_dir}/*.mp4  (512x512 if force_512_video=True)")
    print(f"  - Labels (per frame): {run_dir}/labels/*.txt  (coords match 512x512 frames)")
    if save_frames:
        print(f"  - Per-frame images:   {run_dir}/*.jpg")
    return run_dir
