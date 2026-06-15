# Palm Embedding Migration Design

Date: 2026-06-14

## Goal

Migrate PalmGate from the old classifier-derived embedding flow based on `Palm Recognition.ipynb` to the direct embedding model flow from `Palm Embedding.ipynb`.

The runtime must match the new notebook contract:

```text
MediaPipe ROI -> gray -> CLAHE(2.0, 8x8) -> RGB -> resize 224x224
-> float32 pixels in 0-255 range -> palm_embedding.tflite -> 128-d L2 embedding
```

Existing registered embeddings are incompatible. Users must re-register after this migration.

## Chosen Approach

Use Option A: exact notebook contract.

This means:

- load `palm_embedding.tflite` by default;
- read the model's final TFLite output tensor directly;
- use MediaPipe ROI preprocessing for registration, recognition, and seeding;
- store per-hand enrollment templates;
- compare query embeddings with cosine similarity against stored templates;
- enforce `nim` as the unique user identity field alongside `name`.

## User Identity and Database

`users` should store both a display name and unique student number:

```sql
users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nim TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  embedding BLOB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

Behavior:

- Registration requires `nim` and `name`.
- `nim.strip()` must be non-empty.
- `nim` is a unique string; no numeric-only validation is enforced.
- Duplicate NIM returns HTTP `409`.
- `GET /api/users` returns `id`, `nim`, `name`, and `created_at`.
- Delete still uses internal `id`.
- Logs can keep the existing `matched_name` text field; displaying `nim - name` can be handled by the UI/user list without adding log columns.

Migration:

- Code may add a temporary legacy NIM for old rows if needed, but the real deployment migration is to delete `palmprint.db` and re-register users because embeddings are incompatible.

## Model Loading and Inference

Current code reads an internal 1280-d GlobalAveragePooling tensor from the old classifier model. That must stop.

New behavior:

- `MODEL_PATH` defaults to `palm_embedding.tflite`.
- TFLite is loaded without `experimental_preserve_all_tensors`.
- `_run_inference()` reads `interpreter.get_output_details()[0]`.
- The expected output dimension is `128`.
- The returned embedding is converted to `float32` and L2-normalized defensively.
- Runtime should fail clearly if the model output is not a vector.

If `model_metadata.json` exists next to the model or in the project root, config should use it for:

- `embedding_dim`;
- `operating_threshold`;
- `tta_rotations`;
- documented preprocessing contract.

Default threshold should match the notebook value if metadata is absent:

```text
SIMILARITY_THRESHOLD = 0.745932400226593
```

## Preprocessing

The runtime preprocessing path must match `Palm Embedding.ipynb`, not the old rembg/FFT notebook.

MediaPipe ROI algorithm:

1. Detect one hand.
2. Read wrist, index MCP, middle MCP, and pinky MCP landmarks.
3. Compute `palm_width = norm(index_mcp - pinky_mcp)` in pixels.
4. Reject frames with `palm_width < 40`.
5. Compute angle from index MCP to pinky MCP.
6. Compute crop center as `(wrist + middle_mcp) / 2`.
7. Rotate the frame around that crop center.
8. Crop a square ROI using scale `1.5 * palm_width`.
9. Convert ROI to grayscale.
10. Apply CLAHE with clip limit `2.0` and tile grid `8x8`.
11. Convert back to RGB.
12. Resize to `224x224` and cast to `float32` without dividing by `255`.

`NotebookPreprocessor` can remain in the repository for old-comparison/reference use, but it should no longer be the active registration or recognition path.

`get_embedding_from_notebook_frame()` should either be removed later or become a compatibility wrapper around the MediaPipe path. The migration should avoid a large rename unless needed.

## Browser ROI

The browser currently crops a client-side ROI and sends a rotation angle. For the new contract:

- The browser ROI crop should mirror the same MediaPipe ROI geometry as the server.
- The client should send an already-aligned ROI.
- The server must not rotate that ROI a second time.
- Browser registration must send each captured sample as-is; it should not send one averaged rotation angle for all samples.

A simple implementation is to keep `is_roi=true`, ignore `rotation_angle` for new embeddings, and process the ROI with CLAHE/resize only on the server.

## Registration Flow

Registration captures 5 left-hand samples and 5 right-hand samples.

Enrollment processing:

```text
sample frames
-> 128-d normalized embeddings
-> group by hand
-> average each hand group
-> L2-normalize each hand average
-> store left and right templates
```

Storage:

- `users.embedding` stores an overall fallback average template.
- `user_embeddings` stores one row per hand template:
  - `hand = 'left'`
  - `hand = 'right'`

The current `user_embeddings` table can be reused. No new template table is needed.

Duplicate check:

- Build the candidate left/right templates first.
- Compare each candidate template against existing stored templates.
- Reject registration if any candidate score is greater than or equal to `DUPLICATE_THRESHOLD`.

## Recognition Flow

Recognition processing:

```text
query frame
-> MediaPipe ROI preprocessing
-> 128-d normalized embedding
-> cosine similarity against stored hand templates
-> max score wins
```

Decision:

- If max score >= threshold: `ALLOWED` with matched user.
- Otherwise: `DENIED`, including closest match for diagnostics.

The database read path can continue returning all stored embeddings. After this migration, those rows are templates rather than raw captures.

## TTA

The notebook uses TTA rotations `[0.0, -6.0, 6.0]`.

Runtime config:

```text
ENROLLMENT_TTA_ENABLED=1
RECOGNITION_TTA_ENABLED=0
```

Reason:

- Enrollment TTA improves template stability and is not latency-sensitive.
- Recognition TTA costs about 3x inference time on Orange Pi, so it should be opt-in for production latency testing.
- If thesis evaluation must exactly match notebook metrics, enable recognition TTA and accept the latency.

TTA embedding calculation:

1. Rotate processed ROI by each configured angle.
2. Run model inference for each rotated ROI.
3. L2-normalize each output.
4. Average outputs.
5. L2-normalize the final average.

## API Changes

`POST /api/register` requires:

```json
{
  "nim": "123456789",
  "name": "Naufal",
  "images": [],
  "hands": []
}
```

`POST /api/device-registration/start` requires:

```json
{
  "nim": "123456789",
  "name": "Naufal"
}
```

Validation:

- missing name -> HTTP `400`;
- missing NIM -> HTTP `400`;
- duplicate NIM -> HTTP `409`;
- duplicate palm template -> HTTP `409`.

## UI Changes

Register tab:

- Add a NIM input beside or above the name input.
- Disable start registration until both NIM and name are filled.
- Browser registration sends `nim` to `/api/register`.
- USB registration sends `nim` to `/api/device-registration/start`.

User display:

- User list displays `nim - name`.
- Recognition result can keep showing name only unless the UI already has the NIM available.

## Seed Scripts

Seeding should use the same MediaPipe embedding path as runtime.

Changes:

- Load the hand landmarker for seeding.
- Stop using rembg/FFT preprocessing as the normal seed path.
- Build templates from real multi-image person folders when available.
- For single seed images, use MediaPipe ROI plus TTA; mark this as test-only because one image is weak enrollment.
- Require a NIM source for real seeding. For filename-based seeds, use a documented convention such as `nim_name.jpg` or person folder names containing the NIM.

## Defensive Checks

Because old and new embeddings have different dimensions:

- If stored embedding dimensions do not match the current model embedding dimension, recognition should not crash with a vector shape error.
- The system should return a clear error or skip incompatible rows with a warning telling the operator to delete `palmprint.db` and re-register.

For deployment, the supported migration is still:

```bash
rm palmprint.db
```

then re-register users.

## Documentation and Deployment Updates

Update references from old artifacts and pipeline:

- `README.md`: required model file becomes `palm_embedding.tflite`.
- `README.md`: recognition explanation becomes MediaPipe ROI + 128-d direct embedding.
- `README.md`: registration stores two per-hand templates, not old classifier/GAP embeddings.
- `CLAUDE.md`: project architecture should describe the new embedding notebook contract.
- `docker-compose.yml`: mount `palm_embedding.tflite` instead of `palm_recognition.tflite`.
- deployment notes: copy `palm_embedding.tflite`, `model_metadata.json` if available, and `hand_landmarker.task`.

## Tests

Minimum tests:

- `PalmProcessor` reads final output tensor directly and returns a normalized vector.
- Config defaults to `palm_embedding.tflite` and threshold `0.745932400226593` when metadata is absent.
- Metadata loading overrides threshold and embedding dimension when present.
- Database stores and lists `nim`.
- Duplicate NIM is rejected.
- Browser registration stores two hand templates, not ten raw captures.
- Device registration stores two hand templates, not ten raw captures.
- Recognition compares against templates and returns the max cosine match.
- Incompatible stored embedding dimensions produce a clear failure path.
- Seed service uses the runtime MediaPipe embedding path.

## Deliberate Non-Goals

- No automatic conversion of old embeddings; it is impossible across model/preprocessing changes.
- No new student table; one `users` table is enough.
- No strict numeric validation for NIM; keep it a unique string until the official format is fixed.
- No raw-capture retention in the first migration; add it later only if recalibration without re-registration becomes necessary.
