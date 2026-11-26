# Accessibility Guide (Draft)

This guide explains how to use Offline Local Search with a keyboard, a screen reader, larger text and high contrast. Everything runs on your computer; your files are never uploaded.

## Getting around

The app has three pages, listed in the navigation on the left (or at the bottom in a narrow window):

| Page | What it is for |
| --- | --- |
| Search | Find documents and images |
| Library | Choose which folders are searchable and watch indexing progress |
| Settings | Theme, high contrast, text size, results per search and the service address |

## Keyboard shortcuts

| Keys | Action |
| --- | --- |
| Ctrl+F (Cmd+F on Mac) | Go to the search box |
| Ctrl+1, Ctrl+2, Ctrl+3 | Open Search, Library, Settings |
| Enter | Run the search, or open the focused result |
| Escape | Clear the search box |
| Tab / Shift+Tab | Next / previous control |
| Space | Turn the focused filter or switch on or off |

The control that has keyboard focus always shows a thick outline.

## Searching

1. The search box has focus when the app starts. Type what you remember, for example "budget for model conversion", and press Enter.
2. You hear how many results were found.
3. Press Tab to move through the filters and the Search button, then to the results.
4. Each result is read as one item: its position, file name, type, page, relevance and why it matched (keywords, text meaning or image content). Press Enter to open the file.
5. After a result, Tab reaches its "Show in its folder" button.

**Search methods.** Keywords finds exact words. Text meaning finds documents that say the same thing in other words. Image content finds pictures that match a description such as "a dog on a beach". At least one method always stays on.

**File types.** Choose PDF, Word, Text or Images to narrow the results. With none selected, all types are included.

## Adding folders

1. Press Ctrl+2 for the Library page.
2. Choose **Add folder** and pick a folder in the system dialog.
3. Progress is announced at 25%, 50% and 75%, and a summary is read when indexing finishes.
4. Files that could not be read are listed under **Show … files that could not be read**.
5. Use **Re-index** after changing files, or **Remove** to stop searching a folder. Removing never deletes your files and always asks you to confirm.

## Seeing more comfortably

- **Text size:** Settings → Text size slider, from 100% to 200%. This adds to your operating system's text size.
- **High contrast:** Settings → High contrast switch. The app also switches automatically when Windows contrast themes or macOS Increase Contrast are on.
- **Dark mode:** Settings → Theme → Dark.

## Screen readers

The app has been designed for NVDA and Narrator on Windows and VoiceOver on macOS.

- Page and section titles are headings; in NVDA press H to jump between them, in VoiceOver use the rotor.
- Status messages (search results, indexing progress, errors) are spoken automatically without moving focus.
- The service-not-running banner is announced and has a Retry button.

## When something goes wrong

| Message | What to do |
| --- | --- |
| The local search service is not running | Start the backend service, then choose Retry |
| The search service is starting or its models are missing | Wait a moment, or check the model files with `python tools/verify_models.py` |
| Could not open a file | The file was moved or deleted; re-index its folder |
| Some search methods are unavailable | A model file is missing; the other methods still work |

## Feedback

Please report accessibility problems in the project's GitHub issues, including your operating system, screen reader and the steps that caused the problem.
