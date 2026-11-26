# Week 5 — Flutter UI and Accessibility

**Status:** In progress — implementation and automated checks complete; manual screen-reader, keyboard and usability sessions pending

## 1. Objective

Build the desktop user interface for the Week 4 retrieval service and make every workflow usable with a keyboard, a screen reader, large text and high contrast, targeting WCAG 2.1 AA.

## 2. Implemented Scope

- Search screen: query box, search methods (keywords, text meaning, image content), file-type filters, explained results, open file, show in folder
- Library screen: add, re-index and remove folders; live indexing progress; last-run summary with unreadable files; index statistics
- Settings screen: theme (system, light, dark), high contrast, text size up to 200%, results per search, service address with connection test, keyboard shortcut reference
- Recoverable errors: service-not-running banner with Retry, search and library error panels with Try again, unopenable-file message
- Adaptive navigation: side rail on desktop widths, bottom bar below 600 px
- Settings persisted locally with `shared_preferences`

## 3. Architecture

```text
lib/
├── main.dart                       wiring: settings → HTTP client → app
└── src/
    ├── app.dart                    MaterialApp, themes, text scaling
    ├── platform.dart               folder picker, open file, reveal in folder
    ├── api/
    │   ├── models.dart             typed JSON models
    │   └── retrieval_api.dart      RetrievalApi + HttpRetrievalApi, ApiException
    ├── state/
    │   ├── settings_controller.dart
    │   ├── search_view_model.dart
    │   └── library_view_model.dart (polls indexing jobs)
    └── ui/
        ├── home_shell.dart         navigation, shortcuts, service banner
        ├── search_page.dart
        ├── library_page.dart
        ├── settings_page.dart
        ├── theme.dart              Material 3 + high-contrast palettes
        └── widgets/
            ├── accessibility.dart  announce(), AnnouncedStatus, FocusOutline, ErrorPanel
            └── result_card.dart
```

View models are plain `ChangeNotifier`s that never touch widgets, so they are unit-tested without a UI. Platform integrations are injected through `PlatformActions`, and the service through the `RetrievalApi` interface, so widget tests run against fakes.

Dependencies added (all published by the Flutter or Dart teams, BSD-3-Clause): `http`, `file_selector`, `url_launcher`, `shared_preferences`.

## 4. Accessibility Implementation (WCAG 2.1 AA mapping)

| WCAG criterion | Implementation |
| --- | --- |
| 1.1.1 Non-text content | Icons in buttons have tooltips that become accessible names; thumbnails are excluded from semantics because the result label names the file |
| 1.3.1 Info and relationships | Page and section titles are semantic headers; filter groups have labelled headings; results are grouped in a labelled container |
| 1.4.3 Contrast (minimum) | Theme pairs verified ≥ 4.5:1 by `theme_contrast_test.dart`; rendered text checked by `textContrastGuideline` |
| 1.4.4 Resize text | In-app text size up to 200% on top of the system setting; layouts use `Wrap` and scrolling lists; tested at 200% for overflow |
| 1.4.6 Contrast (enhanced, AAA) | High-contrast themes verified ≥ 7:1; the OS high-contrast setting is honoured automatically |
| 1.4.10 Reflow | Navigation switches layout below 600 px; content wraps instead of scrolling horizontally |
| 1.4.11 Non-text contrast | Focus outline colour ≥ 3:1 against the surface; card borders in high contrast |
| 2.1.1 Keyboard | Every action is reachable with Tab/Shift+Tab, Enter and Space; shortcuts Ctrl+F, Ctrl+1–3, Escape |
| 2.1.2 No keyboard trap | Inactive pages are removed from focus traversal; dialogs return focus on close |
| 2.4.3 Focus order | Reading-order traversal; focus moves into a page after navigation; search box focused at start |
| 2.4.6 Headings and labels | Descriptive headings, field labels and helper text ("Press Enter to search, Escape to clear") |
| 2.4.7 Focus visible | 3 px outline on focused buttons, fields and result cards |
| 2.5.5 Target size (AAA) | 48 × 48 px minimum targets; verified with `androidTapTargetGuideline` and `iOSTapTargetGuideline` |
| 3.3.1 Error identification | Errors say what happened and what to do, with a retry action |
| 3.3.2 Labels or instructions | Every input has a visible label; the service address explains its expected format |
| 4.1.2 Name, role, value | Result cards expose one button with a full description; switches, sliders and chips expose state and value |
| 4.1.3 Status messages | Search outcomes, indexing milestones and errors are announced without moving focus |

## 5. Running the App

```powershell
# Terminal 1: backend (from the repository root, virtual environment active)
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8765

# Terminal 2: frontend
cd frontend
flutter pub get
flutter run -d windows     # or: -d macos
```

The first time on Windows, add the Windows runner with `flutter create --platforms=windows .` inside `frontend/`.

## 6. Automated Tests

```powershell
cd frontend
flutter analyze
flutter test
```

| Test file | What it verifies |
| --- | --- |
| `api_client_test.dart` | Request bodies, JSON parsing, HTTP and connection errors, runtime base-URL changes |
| `view_models_test.dart` | Search filters and status messages, at-least-one-method rule, indexing job following, settings persistence and validation |
| `theme_contrast_test.dart` | Contrast ratios of every theme (4.5:1, 7:1 in high contrast), focus-outline contrast, 48 px targets |
| `app_accessibility_test.dart` | End-to-end widget flows, screen-reader label of results, keyboard shortcuts and Tab order, folder add/remove with confirmation, service-down recovery, Flutter accessibility guidelines (tap targets, labels, text contrast) in default and high-contrast themes, 200% text without overflow |

## 7. Manual Accessibility Validation

Flutter desktop apps are not web pages, so WAVE does not apply and Google Accessibility Scanner only runs on Android. The equivalent desktop tools are **Accessibility Insights for Windows** (FastPass) and the **macOS Accessibility Inspector** audit.

Record each check as Pass, Fail or N/A with a note.

### 7.1 Keyboard only (mouse unused)

Tested 2026-11-26 on Windows 11, Flutter 3.50.1 desktop build, `sample_data` library. The mouse was used only once, to give the app window focus.

| # | Task | Result | Notes |
| --- | --- | --- | --- |
| K1 | Ctrl+2 switches to Library, including after clicking empty space | Pass | Failed before the shortcut fix in section 9 |
| K2 | Tab to "Add folder" and press Enter | Pass | Windows folder picker opens |
| K3 | In the picker: Alt+D, paste the path, Enter, Tab to "Select Folder", Enter | Pass | Indexing progress and summary shown; no duplicate folder row |
| K4 | Tab to the folder's Re-index button and press Enter | Pass | Completion status shown |
| K5 | Ctrl+F | Pass | Search page opens with the cursor in the search box |
| K6 | Type a query, Enter, then Tab through Search button, methods, file types and results; Space toggles a chip | Pass | Focus ring visible at every step |
| K7 | Enter on a result opens the file; Enter on its folder icon reveals it in Explorer | Pass | |
| K8 | Esc in the search box | Pass | Query cleared |
| K9 | Shift+Tab backwards through the whole page | Pass | Reverse order, no keyboard trap, focus never hidden |

### 7.2 Screen readers

| # | Check | NVDA (Windows) | VoiceOver (macOS) |
| --- | --- | --- | --- |
| S1 | Page headings are announced and navigable with H | | |
| S2 | Search box announces its label and hint | | |
| S3 | "N results for …" is announced after a search | | |
| S4 | A result reads file name, type, page, relevance and why it matched | | |
| S5 | Filter chips announce selected / not selected | | |
| S6 | Indexing milestones and completion are announced | | |
| S7 | Service-not-running banner is announced | | |
| S8 | Text-size slider announces its value | | |

### 7.3 Visual

Tested 2026-11-26 on Windows 11.

| # | Check | Result | Notes |
| --- | --- | --- | --- |
| V1 | In-app text size 200% on Search, Library and Settings: no clipped or overlapping text, every page scrolls to the end | Pass | Side navigation overflowed in a short window; fixed (section 9) and re-tested |
| V2 | Narrow window at 200% text: switches to bottom navigation, no horizontal scrolling | Pass | |
| V3 | In-app high contrast (light and dark) and Windows contrast themes | Pass | Text, selected/unselected chips and focus ring clearly distinguishable |
| V4 | Accessibility Insights for Windows FastPass: no failures | | |

## 8. Usability Test Plan (to complete)

**Participants:** 5 people, including at least one screen-reader user and one keyboard-only user if possible.

**Tasks:**

1. Add the folder `sample_data` and wait until indexing finishes.
2. Find the PDF that explains how text is extracted.
3. Find an image by describing it.
4. Show only PDF results for "local documents".
5. Make the text larger and switch to high contrast.
6. Remove a folder from the index.

**Measures:** task success (yes/partial/no), time on task, errors, and the System Usability Scale (SUS) questionnaire after the session. Record observations and quotes, then list issues by severity with fixes.

| Participant | Assistive technology | T1 | T2 | T3 | T4 | T5 | T6 | SUS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | | | | | | | | |

## 9. Findings From Hands-On Testing

| Finding | Change |
| --- | --- |
| On `sample_data`, "local documents without the cloud" showed the relevant PDF at 0% and an unrelated photo at 70%: min-max normalization scores results relative to each other, so the weaker of two text files gets 0 and the only image gets full marks | Calibrated fusion is now the default: keyword coverage and calibrated cosine similarity give absolute scores (backend `ranking.py`, regression test `test_calibrated_scores_do_not_depend_on_other_results`) |
| Ctrl+1/2/3 did nothing after clicking empty space: on desktop that moves focus out of the search field, and the shortcuts only listened to focused widgets | Page shortcuts are registered on `HardwareKeyboard`, so they work wherever focus is (except while a dialog is open). Regression test "page shortcuts work when nothing has focus" |
| At 200% text in a short window the side navigation rail overflowed (yellow and black stripes) | The rail scrolls when it is taller than the window. Regression test "200% text in a short window scrolls instead of overflowing" |
| A search error was shown twice (status line and error panel), so it was also read twice | The status line is announced but not drawn while the error panel is shown. Regression test "a search error is shown once" |
| Service banner and folder rows did not span the page; Search button misaligned | Layout fixes in `home_shell.dart`, `library_page.dart`, `search_page.dart` |

The NQ and COCO benchmarks should be re-run with the calibrated default (`--output-name nq-calibrated`, `coco-calibrated`) and compared with the Week 4 `weighted` results.

## 10. Known Limitations

- The backend must be started separately; launching it from the app is planned for Week 6 integration.
- On macOS the app sandbox limits thumbnail and file access outside user-selected folders; sandbox entitlements will be finalised during Week 8 packaging.
- Announcements use `SemanticsService.announce`; behaviour varies slightly between NVDA, Narrator and VoiceOver and must be confirmed in the manual sessions.
