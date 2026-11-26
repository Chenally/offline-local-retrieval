import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/src/api/retrieval_api.dart';
import 'package:frontend/src/app.dart';
import 'package:frontend/src/state/settings_controller.dart';

import 'support/fakes.dart';

class Harness {
  Harness({bool alive = true, Map<String, Object?>? settings})
      : api = FakeRetrievalApi(alive: alive),
        settings = SettingsController(MemorySettingsStore(settings));

  final FakeRetrievalApi api;
  final SettingsController settings;
  final RecordingPlatform platform = RecordingPlatform();

  Future<void> pump(WidgetTester tester, {Size size = const Size(1280, 900)}) async {
    tester.view.physicalSize = size;
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.reset);

    await settings.load();
    await tester.pumpWidget(
      OfflineRetrievalApp(
        settings: settings,
        api: api,
        platform: platform.actions,
        libraryPollInterval: Duration.zero,
      ),
    );
    await tester.pumpAndSettle();
  }
}

Future<void> pressShortcut(WidgetTester tester, LogicalKeyboardKey key) async {
  await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
  await tester.sendKeyEvent(key);
  await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
  await tester.pumpAndSettle();
}

Future<void> runSearch(WidgetTester tester, String query) async {
  await tester.enterText(find.byType(TextField).first, query);
  await tester.testTextInput.receiveAction(TextInputAction.search);
  await tester.pumpAndSettle();
}

Future<void> expectAccessibleScreen(WidgetTester tester) async {
  await expectLater(tester, meetsGuideline(androidTapTargetGuideline));
  await expectLater(tester, meetsGuideline(iOSTapTargetGuideline));
  await expectLater(tester, meetsGuideline(labeledTapTargetGuideline));
  await expectLater(tester, meetsGuideline(textContrastGuideline));
}

void main() {
  group('Search workflow', () {
    testWidgets('search field has focus on start and Enter runs a search', (tester) async {
      final harness = Harness();
      await harness.pump(tester);

      final field = tester.widget<TextField>(find.byType(TextField).first);
      expect(field.focusNode!.hasFocus, isTrue);

      await runSearch(tester, 'budget');

      expect(harness.api.searchRequests.single.query, 'budget');
      expect(find.text('1 result for "budget".'), findsOneWidget);
      expect(find.text('budget.pdf'), findsOneWidget);
    });

    testWidgets('result exposes one descriptive screen-reader label', (tester) async {
      final handle = tester.ensureSemantics();
      final harness = Harness();
      await harness.pump(tester);
      await runSearch(tester, 'budget');

      expect(
        find.bySemanticsLabel(
          RegExp(r'^Result 1 of 1, budget\.pdf, PDF document, page 2, relevance 87 percent, '
              r'matched by text meaning 80 percent and keywords 100 percent'),
        ),
        findsOneWidget,
      );
      handle.dispose();
    });

    testWidgets('activating a result opens the file; folder button reveals it', (tester) async {
      final harness = Harness();
      await harness.pump(tester);
      await runSearch(tester, 'budget');

      await tester.tap(find.text('budget.pdf'));
      await tester.pumpAndSettle();
      await tester.tap(find.byTooltip('Show budget.pdf in its folder'));
      await tester.pumpAndSettle();

      expect(harness.platform.opened, <String>['/library/budget.pdf']);
      expect(harness.platform.revealed, <String>['/library/budget.pdf']);
    });

    testWidgets('a file that cannot be opened shows a recoverable message', (tester) async {
      final harness = Harness();
      harness.platform.openSucceeds = false;
      await harness.pump(tester);
      await runSearch(tester, 'budget');

      await tester.tap(find.text('budget.pdf'));
      await tester.pumpAndSettle();

      expect(find.textContaining('Could not open budget.pdf'), findsOneWidget);
    });

    testWidgets('Escape clears the search box', (tester) async {
      final harness = Harness();
      await harness.pump(tester);
      await runSearch(tester, 'budget');

      await tester.sendKeyEvent(LogicalKeyboardKey.escape);
      await tester.pumpAndSettle();

      expect(find.text('Search cleared.'), findsOneWidget);
      expect(find.text('budget.pdf'), findsNothing);
    });

    testWidgets('filter chips are toggled from the keyboard state', (tester) async {
      final harness = Harness();
      await harness.pump(tester);

      await tester.tap(find.widgetWithText(FilterChip, 'PDF'));
      await tester.tap(find.widgetWithText(FilterChip, 'Image content'));
      await tester.pumpAndSettle();
      await runSearch(tester, 'budget');

      final request = harness.api.searchRequests.single;
      expect(request.extensions, <String>['pdf']);
      expect(request.modes!.length, 2);
    });
  });

  group('Navigation and keyboard shortcuts', () {
    testWidgets('Ctrl+2 and Ctrl+3 open Library and Settings; Ctrl+F returns to search',
        (tester) async {
      final harness = Harness();
      await harness.pump(tester);

      await pressShortcut(tester, LogicalKeyboardKey.digit2);
      expect(find.text('Index overview'), findsOneWidget);

      await pressShortcut(tester, LogicalKeyboardKey.digit3);
      expect(find.text('Local search service'), findsOneWidget);

      await pressShortcut(tester, LogicalKeyboardKey.keyF);
      final field = tester.widget<TextField>(find.byType(TextField).first);
      expect(field.focusNode!.hasFocus, isTrue);
    });

    testWidgets('page shortcuts work when nothing has focus', (tester) async {
      // Regression: clicking empty space unfocuses the search field, and
      // Ctrl+2 used to do nothing because no focused widget saw the keys.
      final harness = Harness();
      await harness.pump(tester);
      FocusManager.instance.primaryFocus?.unfocus();
      await tester.pump();

      await pressShortcut(tester, LogicalKeyboardKey.digit2);
      expect(find.text('Index overview'), findsOneWidget);
    });

    testWidgets('Tab moves focus from the search box to the Search button', (tester) async {
      final harness = Harness();
      await harness.pump(tester);
      await tester.enterText(find.byType(TextField).first, 'budget');

      await tester.sendKeyEvent(LogicalKeyboardKey.tab);
      await tester.pumpAndSettle();
      await tester.sendKeyEvent(LogicalKeyboardKey.enter);
      await tester.pumpAndSettle();

      expect(harness.api.searchRequests.single.query, 'budget');
    });
  });

  group('Large text', () {
    testWidgets('200% text in a short window scrolls instead of overflowing',
        (tester) async {
      // Regression: the side navigation rail overflowed (yellow and black
      // stripes) at 200% text when the window was short.
      final harness = Harness(settings: <String, Object?>{'textScale': 2.0});
      await harness.pump(tester, size: const Size(1000, 320));
      expect(tester.takeException(), isNull);

      for (final shortcut in <LogicalKeyboardKey>[
        LogicalKeyboardKey.digit2,
        LogicalKeyboardKey.digit3,
      ]) {
        await pressShortcut(tester, shortcut);
        expect(tester.takeException(), isNull);
      }
    });
  });

  group('Errors', () {
    testWidgets('a search error is shown once, in the error panel', (tester) async {
      final harness = Harness();
      harness.api.searchError = const ApiException(ApiErrorKind.unreachable, 'offline');
      await harness.pump(tester);
      await runSearch(tester, 'budget');

      expect(find.textContaining('not responding'), findsOneWidget);
      expect(find.text('Try again'), findsOneWidget);
    });
  });

  group('Library', () {
    testWidgets('adding a folder indexes it and announces the outcome', (tester) async {
      final harness = Harness();
      await harness.pump(tester);
      await pressShortcut(tester, LogicalKeyboardKey.digit2);

      await tester.tap(find.text('Add folder'));
      await tester.pumpAndSettle();

      expect(harness.api.indexedDirectories, <String>['/new-folder']);
      expect(find.textContaining('Finished indexing library'), findsOneWidget);
      expect(find.text('Last indexing result'), findsOneWidget);
    });

    testWidgets('removing a folder asks for confirmation', (tester) async {
      final harness = Harness();
      await harness.pump(tester);
      await pressShortcut(tester, LogicalKeyboardKey.digit2);

      await tester.tap(find.byTooltip('Remove /library from the index'));
      await tester.pumpAndSettle();
      expect(find.text('Remove folder from index?'), findsOneWidget);

      await tester.tap(find.text('Cancel'));
      await tester.pumpAndSettle();
      expect(harness.api.removedDirectories, isEmpty);

      await tester.tap(find.byTooltip('Remove /library from the index'));
      await tester.pumpAndSettle();
      await tester.tap(find.widgetWithText(FilledButton, 'Remove'));
      await tester.pumpAndSettle();
      expect(harness.api.removedDirectories, <String>['/library']);
    });
  });

  group('Error recovery', () {
    testWidgets('unreachable service shows a banner with Retry', (tester) async {
      final harness = Harness(alive: false);
      await harness.pump(tester);

      expect(find.textContaining('not running'), findsOneWidget);

      harness.api.alive = true;
      await tester.tap(find.text('Retry'));
      await tester.pumpAndSettle();

      expect(find.textContaining('not running'), findsNothing);
    });
  });

  group('Accessibility guidelines', () {
    for (final highContrast in <bool>[false, true]) {
      final mode = highContrast ? 'high contrast' : 'default theme';

      testWidgets('search results meet tap-target, label and contrast guidelines ($mode)',
          (tester) async {
        final handle = tester.ensureSemantics();
        final harness = Harness(settings: <String, Object?>{'highContrast': highContrast});
        await harness.pump(tester);
        await runSearch(tester, 'budget');

        await expectAccessibleScreen(tester);
        handle.dispose();
      });

      testWidgets('library page meets guidelines ($mode)', (tester) async {
        final handle = tester.ensureSemantics();
        final harness = Harness(settings: <String, Object?>{'highContrast': highContrast});
        await harness.pump(tester);
        await pressShortcut(tester, LogicalKeyboardKey.digit2);

        await expectAccessibleScreen(tester);
        handle.dispose();
      });

      testWidgets('settings page meets guidelines ($mode)', (tester) async {
        final handle = tester.ensureSemantics();
        final harness = Harness(settings: <String, Object?>{'highContrast': highContrast});
        await harness.pump(tester);
        await pressShortcut(tester, LogicalKeyboardKey.digit3);

        await expectAccessibleScreen(tester);
        handle.dispose();
      });
    }

    for (final size in <Size>[const Size(1280, 900), const Size(800, 900)]) {
      testWidgets('200% text reflows without overflow at ${size.width.toInt()} px wide',
          (tester) async {
        final harness = Harness(settings: <String, Object?>{'textScale': 2.0});
        await harness.pump(tester, size: size);
        await runSearch(tester, 'budget');

        for (final shortcut in <LogicalKeyboardKey>[
          LogicalKeyboardKey.digit2,
          LogicalKeyboardKey.digit3,
        ]) {
          await pressShortcut(tester, shortcut);
        }

        expect(tester.takeException(), isNull);
      });
    }
  });
}
