// Diagnostic logging for the Windows screen reader crash.
//
// Enabled only with `--dart-define=A11Y_DEBUG=true`. After every semantics
// update it compares each node's parent with the previous update and prints
// nodes that changed parent, marking moves away from the root (which the
// Windows accessibility bridge cannot apply). It also prints window size
// changes and the top of the tree when semantics first turn on.

import 'package:flutter/rendering.dart';
import 'package:flutter/scheduler.dart';

const bool kA11yDebug = bool.fromEnvironment('A11Y_DEBUG');

class SemanticsLogger {
  SemanticsLogger._();

  static SemanticsOwner? _owner;
  static Map<int, int> _parents = <int, int>{};
  static Size? _lastSize;
  static int _update = 0;

  static void install() {
    SchedulerBinding.instance.addPersistentFrameCallback((_) => _onFrame());
    debugPrint('[a11y] semantics logger installed');
  }

  static void _onFrame() {
    final binding = RendererBinding.instance;

    for (final view in binding.renderViews) {
      final size = view.flutterView.physicalSize;
      if (size != _lastSize) {
        debugPrint('[a11y] window size ${size.width}x${size.height}');
        _lastSize = size;
      }

      final owner = view.owner?.semanticsOwner;
      if (owner != null && !identical(owner, _owner)) {
        _owner?.removeListener(_onUpdate);
        _owner = owner;
        _parents = <int, int>{};
        owner.addListener(_onUpdate);
        debugPrint('[a11y] semantics enabled');
      }
      break;
    }
  }

  static void _onUpdate() {
    final root = _owner?.rootSemanticsNode;
    if (root == null) {
      return;
    }
    _update++;

    final parents = <int, int>{};
    final nodes = <int, SemanticsNode>{root.id: root};

    void walk(SemanticsNode node) {
      node.visitChildren((child) {
        parents[child.id] = node.id;
        nodes[child.id] = child;
        walk(child);
        return true;
      });
    }

    walk(root);

    if (_update <= 2) {
      debugPrint('[a11y] update $_update, tree:\n${_top(root)}');
    }

    // The Windows bridge builds its tree from each node's children in
    // traversal order only. A node that no traversal list reaches is never
    // attached ("Nodes left pending by the update").
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
        final parent = nodes[parents[node.id]];
        debugPrint('[a11y] ORPHAN in update $_update: ${_describe(node)} '
            'hit-test parent ${parent == null ? '?' : _describe(parent)}');
      }
    }

    for (final entry in parents.entries) {
      final before = _parents[entry.key];
      if (before != null && before != entry.value) {
        final marker = before == root.id ? 'ROOT MOVE' : 'move';
        debugPrint(
          '[a11y] $marker in update $_update: ${_describe(nodes[entry.key]!)} '
          'parent $before -> ${entry.value} ${_describe(nodes[entry.value]!)}',
        );
      }
    }

    final rootChildren = <int>[];
    root.visitChildren((child) {
      rootChildren.add(child.id);
      return true;
    });
    final previousRootChildren = _parents.entries
        .where((entry) => entry.value == root.id)
        .map((entry) => entry.key)
        .toList();
    if (previousRootChildren.join(',') != rootChildren.join(',')) {
      debugPrint('[a11y] update $_update: root children '
          '$previousRootChildren -> $rootChildren');
    }

    _parents = parents;
  }

  static String _top(SemanticsNode root, {int depth = 12}) {
    final lines = <String>[];
    void walk(SemanticsNode node, int level) {
      lines.add('${'  ' * level}${_describe(node)}');
      if (level < depth) {
        node.visitChildren((child) {
          walk(child, level + 1);
          return true;
        });
      }
    }

    walk(root, 0);
    return lines.join('\n');
  }

  static String _describe(SemanticsNode node) {
    final data = node.getSemanticsData();
    final children = <int>[];
    node.visitChildren((child) {
      children.add(child.id);
      return true;
    });
    final text = <String>[
      if (data.label.isNotEmpty) 'label="${data.label}"',
      if (data.value.isNotEmpty) 'value="${data.value}"',
      if (data.hint.isNotEmpty) 'hint="${data.hint}"',
    ].join(' ');
    final rect = node.rect;
    return '#${node.id} $text '
        'rect=(${rect.left.round()},${rect.top.round()} '
        '${rect.width.round()}x${rect.height.round()}) children=$children';
  }
}
