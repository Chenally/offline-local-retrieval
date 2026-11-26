# Offline Local Search — Flutter desktop app

Accessible desktop interface for the local retrieval service in `../backend`.

## Run

1. Start the backend from the repository root:

   ```bash
   python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8765
   ```

2. Start the app:

   ```bash
   flutter pub get
   flutter run -d windows   # or -d macos / -d linux
   ```

   On a new machine, add the platform runner first, for example `flutter create --platforms=windows .`

## Test

```bash
flutter analyze
flutter test
```

See `../docs/week5/README.md` for architecture, the WCAG 2.1 AA mapping and the manual test protocol, and `../docs/accessibility_guide.md` for the user guide.
