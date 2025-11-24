# Face Recognition System Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Real-Time Face Recognition System                   │
└─────────────────────────────────────────────────────────────────────────┘

                                    ┌──────────┐
                                    │  Webcam  │
                                    └────┬─────┘
                                         │ Video Stream
                                         ▼
                            ┌────────────────────────┐
                            │   Frame Capture Loop   │
                            │    (OpenCV Camera)     │
                            └───────────┬────────────┘
                                        │ Raw Frame (BGR)
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          DETECTION PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                    Face Detector (detector.py)                  │     │
│  │  ┌──────────────────────────────────────────────────────────┐   │     │
│  │  │              MTCNN (facenet-pytorch)                     │   │     │
│  │  │  - Device: MPS/CPU (adaptive)                            │   │     │
│  │  │  - Image padding for MPS (pad_mult=64)                   │   │     │
│  │  │  - CPU fallback on MPS errors                            │   │     │
│  │  └──────────────────────────────────────────────────────────┘   │     │
│  │                           ↓                                     │     │
│  │         Outputs: Bounding Boxes, Probabilities,                 │     │
│  │                  Aligned Face Tensors (Nx3x160x160)             │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │ Every N frames (--detect-every)       │
                    └───────────────────┬───────────────────┘
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                          EMBEDDING PIPELINE                              │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                   Embedder (embedder.py)                        │     │
│  │  ┌──────────────────────────────────────────────────────────┐   │     │
│  │  │  Embedding Model (Device: MPS/CPU)                       │   │     │
│  │  │  ┌────────────────────┬──────────────────────────────┐   │   │     │
│  │  │  │ InceptionResnetV1  │    MobileFaceNet (Custom)    │   │   │     │
│  │  │  │  (vggface2)        │      (Lightweight)           │   │   │     │
│  │  │  │  512-d embeddings  │    512-d embeddings          │   │   │     │
│  │  │  └────────────────────┴──────────────────────────────┘   │   │     │
│  │  └──────────────────────────────────────────────────────────┘   │     │
│  │                           ↓                                     │     │
│  │              L2-Normalized Embeddings (Nx512)                   │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                         RECOGNITION PIPELINE                             │
├──────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                  Recognizer (recognizer.py)                     │     │
│  │  ┌──────────────────────────────────────────────────────────┐   │     │
│  │  │          Face Database (face_db.pkl)                     │   │     │
│  │  │  {                                                       │   │     │
│  │  │    "person_id": {                                        │   │     │
│  │  │      "name": "Koushik",                                  │   │     │
│  │  │      "embedding": [512-d vector],                        │   │     │
│  │  │      "imgs": [paths]                                     │   │     │
│  │  │    }                                                     │   │     │
│  │  │  }                                                       │   │     │
│  │  └──────────────────────────────────────────────────────────┘   │     │
│  │                           ↓                                     │     │
│  │         Cosine Similarity Matching (threshold=0.6)              │     │
│  │                           ↓                                     │     │
│  │            Outputs: (Name, Similarity Score)                    │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRACKING PIPELINE                              │
├───────────────────────────────────────────────────────────── ────────────┤
│  ┌─────────────────────────────────────────────────────────────────┐     │
│  │                    Tracker (tracker.py)                         │     │
│  │  ┌──────────────────────────────────────────────────────────┐   │     │
│  │  │           Simple IOU Tracker with EMA Smoothing          │   │     │
│  │  │  - Assigns unique Track IDs                              │   │     │
│  │  │  - Smooths bounding boxes (α=0.6)                        │   │     │
│  │  │  - Persists labels/scores per track                      │   │     │
│  │  └──────────────────────────────────────────────────────────┘   │     │
│  │                           ↓                                     │     │
│  │          Stable Track IDs + Smoothed Boxes + Labels             │     │
│  └─────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    ▼                                       ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │  Visual Output        │             │  Audio Output (TTS)   │
        │  (utils.py)           │             │  (macOS 'say')        │
        ├───────────────────────┤             ├───────────────────────┤
        │ • Draw bounding boxes │             │ • Greeting on detect  │
        │ • Display name + score│             │ • "Hello Doctor..."   │
        │ • Show FPS counter    │             │ • Cooldown interval   │
        │ • OpenCV window       │             │ • Pronunciation fixes │
        └───────────────────────┘             └───────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        ▼
                            ┌───────────────────────┐
                            │   CSV Logger          │
                            │ (recognitions.csv)    │
                            │ timestamp, track_id,  │
                            │ name, score           │
                            └───────────────────────┘
```

---

## Component Details

### 1. **Face Detection** (`detector.py`)
- **Technology**: MTCNN from facenet-pytorch
- **Device Strategy**: 
  - Primary: MPS (Apple Silicon GPU)
  - Fallback: CPU (on MPS errors)
- **Key Features**:
  - Adaptive padding for MPS compatibility
  - Returns aligned face tensors (160×160)
  - Throttled warning system

### 2. **Embedding Extraction** (`embedder.py`)
- **Models Available**:
  - **InceptionResnetV1** (default): VGGFace2 pretrained, 512-d
  - **MobileFaceNet** (optional): Lightweight custom model
- **Device**: MPS or CPU
- **Output**: L2-normalized 512-dimensional vectors

### 3. **Face Recognition** (`recognizer.py`)
- **Method**: Cosine similarity matching
- **Database**: Pickle file mapping person_id → embeddings
- **Threshold**: Configurable (default 0.6)

### 4. **Tracking** (`tracker.py`)
- **Algorithm**: IOU-based tracking
- **Features**:
  - Exponential Moving Average (EMA) box smoothing
  - Per-track label/score persistence
  - Reduces flicker and improves stability

### 5. **Real-time Demo** (`realtime.py`)
- **Orchestration**: Main loop coordinating all components
- **Optimization**: Detect every N frames (default: 3)
- **Output**: Visual display + TTS + CSV logging

---

## Data Flow

```
Camera → Detect → Embed → Match → Track → Display/Speak/Log
         (MTCNN)  (ResNet) (Cosine) (IOU)   (OpenCV/say/CSV)
```

---

## Database Building Pipeline

```
┌──────────────────┐
│  Campus Faces    │
│  Directory       │
│  ├── Koushik/    │
│  ├── Kartik/     │
│  └── Dr. .../    │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│  database.py             │
│  1. Load images          │
│  2. Detect faces (CPU)   │
│  3. Compute embeddings   │
│  4. Average per person   │
│  5. Save to pickle       │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│   face_db.pkl            │
│   {person: embedding}    │
└──────────────────────────┘
```

---

## Enrollment Flow

```
┌─────────────┐
│  enroll.py  │
│  - Capture  │
│  - Save     │
└──────┬──────┘
       │
       ▼
┌──────────────────┐
│ campus_faces/    │
│   <person_id>/   │
│     img1.jpg     │
│     img2.jpg     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  Rebuild DB      │
│  database.py     │
└──────────────────┘
```

---

## Hardware Optimization (Apple M4)

- **MPS Backend**: PyTorch Metal Performance Shaders
- **Fallback Strategy**: Automatic CPU fallback on errors
- **Memory**: Efficient tensor management
- **Performance**: ~20-30 FPS on M4

---

## Configuration Options

| Flag | Purpose | Default |
|------|---------|---------|
| `--device` | Embedding device | `mps` |
| `--detection-device` | Detection device | Same as device |
| `--embedder` | Model choice | `resnet` |
| `--detect-every` | Detection interval | `3` |
| `--threshold` | Recognition threshold | `0.6` |
| `--greet` | Enable TTS | `False` |
| `--smoothing-alpha` | Tracker smoothing | `0.6` |
| `--pad-mult` | MPS padding multiple | `64` |

---

## Key Files

```
face_recognition_m4/
├── src/
│   ├── detector.py          # MTCNN wrapper
│   ├── embedder.py          # InceptionResnet/MobileFaceNet
│   ├── recognizer.py        # Cosine similarity matching
│   ├── tracker.py           # IOU tracking with smoothing
│   ├── database.py          # DB builder
│   ├── realtime.py          # Main demo
│   ├── enroll.py            # Enrollment helper
│   └── utils.py             # Drawing/resizing utilities
├── campus_faces/            # Training images
├── face_db.pkl              # Embedding database
├── recognitions.csv         # Log file
└── requirements-m4.txt      # Dependencies
```

---

## Performance Characteristics

- **Detection**: ~50-100ms per frame (MPS)
- **Embedding**: ~10-20ms per face (MPS)
- **Matching**: <1ms per face
- **Overall FPS**: 20-30 FPS (depending on face count)

---

*Generated for Apple Silicon M4 optimized real-time face recognition system*
