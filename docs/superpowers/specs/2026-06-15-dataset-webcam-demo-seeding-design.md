# Dataset Webcam Demo Seeding Design

Date: 2026-06-15

## Goal

Add a testing-only seed mode for `Dataset_Webcam/`, whose folders are plain person names (`Afrizal`, `Naufal`, `Reza`, `Rizky`) instead of production `nim_name` labels.

## Chosen Approach

Add a CLI flag to the existing seed script:

```bash
python scripts/seed_users.py Dataset_Webcam --replace-users --auto-demo-nim
```

This keeps production seeding strict by default while making the existing test dataset usable without renaming folders.

## Behavior

When `--auto-demo-nim` is enabled:

- Folder names without `nim_name` format are accepted.
- Stable demo NIMs are generated from sorted folder order:
  - `SEED-001`
  - `SEED-002`
  - `SEED-003`
  - `SEED-004`
- The folder name remains the user display name.
- All valid images in the folder are embedded with the current runtime flow:
  - MediaPipe ROI
  - CLAHE preprocessing
  - `palm_embedding.tflite`
  - TTA enabled for seeding
- The valid embeddings are averaged into one normalized template.
- The template is stored as one `user_embeddings` row with `hand = "unknown"`.

## Production Safety

Default behavior stays strict:

- Without `--auto-demo-nim`, labels must still use `nim_name` format.
- The auto-generated `SEED-xxx` NIMs are only created when the explicit flag is passed.
- No fake left/right templates are created.

## Error Handling

- If a folder has no valid MediaPipe embeddings, it is reported under `FAILED`.
- If `--replace-users` is passed, existing users are removed and logs are preserved through the existing delete behavior.
- Duplicate demo NIM collisions are handled by the existing database uniqueness check.

## Tests

Add tests for:

- plain folder labels fail without `--auto-demo-nim`;
- plain folder labels seed successfully with stable `SEED-001`, `SEED-002`, etc.;
- demo seed entries store one `unknown` hand template;
- `scripts/seed_users.py` exposes the `--auto-demo-nim` option.
