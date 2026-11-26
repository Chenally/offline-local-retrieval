// Diagnostic for the Windows screen reader crash.
//
// With NVDA running, the Windows accessibility bridge logged
// "Failed to update ui::AXTree, error: 0 will not be in the tree and is not
// the new root" and the app closed. The bridge fails when a semantics node
// moves away from the root node (id 0) to another parent. This test walks
// the semantics tree after every frame of common interactions and reports
// every node that changes parent, so the widget responsible can be found.

import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/src/app.dart';
import 'package:frontend/src/state/settings_controller.dart';
import 'package:material_ui/material_ui.dart';

import 'support/fakes.dart';

class _Tracker {
  _Tracker(this.tester);

  final WidgetTester tester;
  SemanticsHandle? handle;
  Map<int, int> _parents = <int, int>{};
  final List<String> moves = <String>[];
  final List<String> rootMoves = <String>[];
  final List<String> orderMismatches = <String>[];

  SemanticsNode _root() => _rootOf(tester);

  void check(String step) {
    final root = _root();
    final parents = <int, int>{};
    final nodes = <int, SemanticsNode>{};

    void walk(SemanticsNode node) {
      node.visitChildren((child) {
        parents[child.id] = node.id;
        nodes[child.id] = child;
        walk(child);
        return true;
      });
    }

    walk(root);

    final reached = <int>{root.id};
    void reach(SemanticsNode node) {
      for (final child
          in node.debugListChildrenInOrder(DebugSemanticsDumpOrder.traversalOrder)) {
        if (reached.add(child.id)) {
          reach(child);
        }
      }
    }

    reach(root);
    for (final node in nodes.values) {
      if (!reached.contains(node.id)) {
        final line = '[$step] ${_describe(node)} is not reachable in traversal '
            'order (hit-test parent ${parents[node.id]})';
        orderMismatches.add(line);
      }
    }

    for (final entry in parents.entries) {
      final before = _parents[entry.key];
      if (before != null && before != entry.value) {
        final node = nodes[entry.key]!;
        final line = '[$step] node ${entry.key} moved from parent $before to '
            '${entry.value}: ${_describe(node)}\n'
            '    new parent: ${_describe(nodes[entry.value] ?? root)}';
        moves.add(line);
        if (before == root.id) {
          rootMoves.add(line);
        }
      }
    }
    _parents = parents;
  }

  static String _describe(SemanticsNode node) {
    final data = node.getSemanticsData();
    final label = data.label.isEmpty ? '' : ' label="${data.label}"';
    final children = <int>[];
    node.visitChildren((child) {
      children.add(child.id);
      return true;
    });
    return 'SemanticsNode#${node.id}$label rect=${node.rect} '
        'children=$children';
  }

  Future<void> frames(String step, {int count = 6}) async {
    for (var i = 0; i < count; i++) {
      await tester.pump(const Duration(milliseconds: 50));
      check('$step, frame $i');
    }
    await tester.pumpAndSettle();
    check('$step, settled');
  }
}

Future<void> _shortcut(WidgetTester tester, LogicalKeyboardKey key) async {
  await tester.sendKeyDownEvent(LogicalKeyboardKey.controlLeft);
  await tester.sendKeyEvent(key);
  await tester.sendKeyUpEvent(LogicalKeyboardKey.controlLeft);
}

SemanticsNode _rootOf(WidgetTester tester) {
  for (final view in tester.binding.renderViews) {
    final root = view.owner?.semanticsOwner?.rootSemanticsNode;
    if (root != null) {
      return root;
    }
  }
  throw StateError('Semantics are not enabled');
}

String _topOfTree(WidgetTester tester, {int depth = 4}) {
  final lines = <String>[];
  void walk(SemanticsNode node, int level) {
    lines.add('${'  ' * level}${_Tracker._describe(node)}');
    if (level < depth) {
      node.visitChildren((child) {
        walk(child, level + 1);
        return true;
      });
    }
  }

  walk(_rootOf(tester), 0);
  return lines.join('\n');
}

/// A window with no size cannot lay out without overflow errors; those are
/// expected here and not what this test checks.
Future<void> _ignoringLayoutErrors(Future<void> Function() body) async {
  final original = FlutterError.onError;
  FlutterError.onError = (details) {};
  try {
    await body();
  } finally {
    FlutterError.onError = original;
  }
}

Future<_Tracker> _launch(
  WidgetTester tester, {
  required Size initialSize,
  required bool semanticsFirst,
}) async {
  tester.view.physicalSize = initialSize;
  tester.view.devicePixelRatio = 1.0;
  addTearDown(tester.view.reset);

  Future<void> guarded(Future<void> Function() body) =>
      initialSize.isEmpty ? _ignoringLayoutErrors(body) : body();

  SemanticsHandle? handle;
  if (semanticsFirst) {
    handle = tester.ensureSemantics();
  }

  final settings = SettingsController(MemorySettingsStore(null));
  await settings.load();

  final app = OfflineRetrievalApp(
    settings: settings,
    api: FakeRetrievalApi(),
    platform: RecordingPlatform().actions,
    libraryPollInterval: Duration.zero,
  );

  await guarded(() async {
    await tester.pumpWidget(app);
    if (!semanticsFirst) {
      // A screen reader that starts after the app (or is detected late).
      await tester.pumpAndSettle();
      handle = tester.ensureSemantics();
      await tester.pump();
    }
  });

  final tracker = _Tracker(tester)..handle = handle;
  tracker.check('first frame with semantics');

  if (initialSize.isEmpty) {
    await _ignoringLayoutErrors(() async {
      await tester.pump(const Duration(milliseconds: 50));
      tracker.check('still no size');
      tester.view.physicalSize = const Size(1280, 720);
      await tester.pump();
    });
    tracker.check('window shown at full size');
  }
  return tracker;
}

Future<void> _exercise(WidgetTester tester, _Tracker tracker) async {
  await tracker.frames('startup');

  // Taller, shorter, wider and narrower windows (crossing the 600 px
  // breakpoint), like dragging the window edges.
  for (final size in <Size>[
    const Size(1280, 1000),
    const Size(1280, 300),
    const Size(1600, 900),
    const Size(500, 900),
    const Size(1280, 720),
  ]) {
    tester.view.physicalSize = size;
    await tracker.frames('resize to ${size.width.toInt()}x${size.height.toInt()}');
  }

  await tester.enterText(find.byType(TextField).first, 'local documents');
  await tester.testTextInput.receiveAction(TextInputAction.search);
  await tracker.frames('search');

  await tester.drag(
    find.byType(Scrollable).last,
    const Offset(0, -300),
    warnIfMissed: false,
  );
  await tracker.frames('scroll');

  for (final key in <LogicalKeyboardKey>[
    LogicalKeyboardKey.digit2,
    LogicalKeyboardKey.digit3,
    LogicalKeyboardKey.digit1,
  ]) {
    await _shortcut(tester, key);
    await tracker.frames('Ctrl+${key.keyLabel}');
  }

  await tester.sendKeyEvent(LogicalKeyboardKey.tab);
  await tracker.frames('Tab');

  tester.view.physicalSize = const Size(1280, 300);
  await tracker.frames('short window after search');

  // Minimize and restore.
  await _ignoringLayoutErrors(() async {
    tester.view.physicalSize = Size.zero;
    await tester.pump();
    tracker.check('minimized');
    await tester.pump(const Duration(milliseconds: 50));
    tracker.check('minimized, later frame');
    tester.view.physicalSize = const Size(1280, 720);
    await tester.pump();
  });
  await tracker.frames('restored');
}

void _report(String scenario, _Tracker tracker) {
  debugPrint('---- $scenario: ${tracker.moves.length} parent changes '
      '(${tracker.rootMoves.length} away from the root) ----');
  for (final line in tracker.moves) {
    debugPrint(line);
  }
  debugPrint('---- $scenario: ${tracker.orderMismatches.length} nodes '
      'not reachable in traversal order ----');
  for (final line in tracker.orderMismatches.take(20)) {
    debugPrint(line);
  }
}

void main() {
  final scenarios = <String, (Size, bool)>{
    'semantics on from launch': (const Size(1280, 720), true),
    'screen reader detected after launch': (const Size(1280, 720), false),
    'window starts small, then grows': (const Size(400, 300), true),
    'window starts with no size, then is shown': (Size.zero, true),
    'no size and screen reader detected after launch': (Size.zero, false),
  };

  for (final MapEntry(key: name, value: (size, semanticsFirst)) in scenarios.entries) {
    testWidgets(
      'no semantics node changes parent: $name',
      variant: TargetPlatformVariant.only(TargetPlatform.windows),
      (tester) async {
        final tracker = await _launch(
          tester,
          initialSize: size,
          semanticsFirst: semanticsFirst,
        );
        if (name == 'semantics on from launch') {
          debugPrint('---- top of the semantics tree ----\n${_topOfTree(tester)}');
        }
        tester.view.physicalSize = const Size(1280, 720);
        await _exercise(tester, tracker);
        _report(name, tracker);
        tracker.handle?.dispose();

        // Any move can land on the root in the real window, where the
        // Windows accessibility bridge cannot apply it.
        expect(tracker.moves, isEmpty, reason: 'semantics nodes changed parent');
        expect(
          tracker.orderMismatches,
          isEmpty,
          reason: 'children missing from traversal order are never attached '
              'by the Windows accessibility bridge',
        );
      },
    );
  }
}
