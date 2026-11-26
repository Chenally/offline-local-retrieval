import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

import '../api/retrieval_api.dart';
import '../platform.dart';
import '../state/library_view_model.dart';
import '../state/search_view_model.dart';
import '../state/settings_controller.dart';
import 'library_page.dart';
import 'search_page.dart';
import 'settings_page.dart';
import 'widgets/accessibility.dart';

/// Width at which navigation moves from a bottom bar to a side rail.
const double wideLayoutBreakpoint = 600;

class HomeShell extends StatefulWidget {
  const HomeShell({
    super.key,
    required this.api,
    required this.settings,
    required this.search,
    required this.library,
    required this.platform,
  });

  final RetrievalApi api;
  final SettingsController settings;
  final SearchViewModel search;
  final LibraryViewModel library;
  final PlatformActions platform;

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  final FocusNode _searchFocus = FocusNode(debugLabel: 'search field');

  /// One programmatic focus target per page, so focus moves into the newly
  /// selected page (and keyboard shortcuts keep working) after navigation.
  final List<FocusNode> _pageFocus = List<FocusNode>.generate(
    3,
    (index) => FocusNode(debugLabel: 'page $index', skipTraversal: true),
  );
  int _index = 0;
  bool? _serviceReachable;

  static const List<_Destination> _destinations = <_Destination>[
    _Destination('Search', Icons.search_outlined, Icons.search),
    _Destination('Library', Icons.folder_outlined, Icons.folder),
    _Destination('Settings', Icons.settings_outlined, Icons.settings),
  ];

  @override
  void initState() {
    super.initState();
    // Registered on HardwareKeyboard rather than as CallbackShortcuts so the
    // page shortcuts work wherever focus is, including when nothing inside
    // the shell has focus (for example after clicking empty space, which
    // moves focus out of a text field on desktop).
    HardwareKeyboard.instance.addHandler(_handleShortcut);
    _checkService();
  }

  @override
  void dispose() {
    HardwareKeyboard.instance.removeHandler(_handleShortcut);
    _searchFocus.dispose();
    for (final node in _pageFocus) {
      node.dispose();
    }
    super.dispose();
  }

  Future<void> _checkService() async {
    final reachable = await widget.api.isAlive();
    if (!mounted) {
      return;
    }
    final changed = reachable != _serviceReachable;
    setState(() => _serviceReachable = reachable);

    if (reachable) {
      await widget.library.refresh();
    } else if (changed && mounted) {
      announce(context, 'The local search service is not running.');
    }
  }

  void _onServiceUrlChanged() {
    widget.api.baseUri = Uri.parse(widget.settings.value.serviceUrl);
    _checkService();
  }

  /// Ctrl+F / Ctrl+1 → Search, Ctrl+2 → Library, Ctrl+3 → Settings
  /// (Cmd instead of Ctrl on macOS).
  bool _handleShortcut(KeyEvent event) {
    if (event is! KeyDownEvent || !mounted) {
      return false;
    }

    final keyboard = HardwareKeyboard.instance;
    final command = keyboard.isControlPressed || keyboard.isMetaPressed;

    if (!command || keyboard.isAltPressed || keyboard.isShiftPressed) {
      return false;
    }

    // Leave shortcuts alone while a dialog is open above the shell.
    final route = ModalRoute.of(context);
    if (route != null && !route.isCurrent) {
      return false;
    }

    final key = event.logicalKey;
    final int target;

    if (key == LogicalKeyboardKey.keyF ||
        key == LogicalKeyboardKey.digit1 ||
        key == LogicalKeyboardKey.numpad1) {
      target = 0;
    } else if (key == LogicalKeyboardKey.digit2 || key == LogicalKeyboardKey.numpad2) {
      target = 1;
    } else if (key == LogicalKeyboardKey.digit3 || key == LogicalKeyboardKey.numpad3) {
      target = 2;
    } else {
      return false;
    }

    _select(target);
    return true;
  }

  void _select(int index) {
    // A tooltip left open would keep a semantics node whose anchor is on
    // the page being hidden.
    Tooltip.dismissAllToolTips();
    setState(() => _index = index);
    announce(context, '${_destinations[index].label} page');
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (!mounted) {
        return;
      }
      if (index == 0) {
        _searchFocus.requestFocus();
      } else {
        _pageFocus[index].requestFocus();
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final pages = <Widget>[
      SearchPage(
        viewModel: widget.search,
        settings: widget.settings,
        platform: widget.platform,
        searchFocusNode: _searchFocus,
      ),
      LibraryPage(viewModel: widget.library, platform: widget.platform),
      SettingsPage(
        settings: widget.settings,
        api: widget.api,
        onServiceUrlChanged: _onServiceUrlChanged,
      ),
    ];

    final body = Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: <Widget>[
        if (_serviceReachable == false)
          _ServiceBanner(
            serviceUrl: widget.settings.value.serviceUrl,
            onRetry: _checkService,
          ),
        Expanded(
          child: IndexedStack(
            index: _index,
            children: <Widget>[
              for (var i = 0; i < pages.length; i++)
                ExcludeFocus(
                  excluding: i != _index,
                  // Hide inactive pages from keyboard focus and screen readers.
                  child: ExcludeSemantics(
                    excluding: i != _index,
                    // A programmatic focus target only. It must not add a
                    // semantics node: toggling its "focused" state inserted
                    // a node above the page, moving the page's node to a new
                    // parent, which the Windows accessibility bridge can fail
                    // to apply (NVDA crash, see docs/week5 section 9).
                    child: Focus(
                      focusNode: _pageFocus[i],
                      includeSemantics: false,
                      child: pages[i],
                    ),
                  ),
                ),
            ],
          ),
        ),
      ],
    );

    return Focus(
      includeSemantics: false,
      child: LayoutBuilder(
        builder: (context, constraints) {
          final wide = constraints.maxWidth >= wideLayoutBreakpoint;

          if (wide) {
            return Scaffold(
              body: SafeArea(
                child: Row(
                  children: <Widget>[
                    _ScrollableRail(
                      child: NavigationRail(
                        selectedIndex: _index,
                        onDestinationSelected: _select,
                        labelType: NavigationRailLabelType.all,
                        leading: const Padding(
                          padding: EdgeInsets.symmetric(vertical: 16),
                          child: _AppMark(),
                        ),
                        destinations: <NavigationRailDestination>[
                          for (final destination in _destinations)
                            NavigationRailDestination(
                              icon: Icon(destination.icon),
                              selectedIcon: Icon(destination.selectedIcon),
                              label: Text(destination.label),
                            ),
                        ],
                      ),
                    ),
                    const VerticalDivider(width: 1),
                    Expanded(child: body),
                  ],
                ),
              ),
            );
          }

          return Scaffold(
            appBar: AppBar(title: const Text('Local Search')),
            body: SafeArea(child: body),
            bottomNavigationBar: NavigationBar(
              selectedIndex: _index,
              onDestinationSelected: _select,
              destinations: <NavigationDestination>[
                for (final destination in _destinations)
                  NavigationDestination(
                    icon: Icon(destination.icon),
                    selectedIcon: Icon(destination.selectedIcon),
                    label: destination.label,
                  ),
              ],
            ),
          );
        },
      ),
    );
  }
}

/// Lets the side rail scroll when large text or a short window makes it
/// taller than the window, instead of overflowing (WCAG 1.4.4, 1.4.10).
class _ScrollableRail extends StatelessWidget {
  const _ScrollableRail({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) => SingleChildScrollView(
        child: ConstrainedBox(
          constraints: BoxConstraints(minHeight: constraints.maxHeight),
          child: IntrinsicHeight(child: child),
        ),
      ),
    );
  }
}

class _Destination {
  const _Destination(this.label, this.icon, this.selectedIcon);

  final String label;
  final IconData icon;
  final IconData selectedIcon;
}

class _AppMark extends StatelessWidget {
  const _AppMark();

  @override
  Widget build(BuildContext context) {
    return Semantics(
      label: 'Offline Local Search',
      header: true,
      excludeSemantics: true,
      child: Icon(
        Icons.travel_explore,
        size: 32,
        color: Theme.of(context).colorScheme.primary,
      ),
    );
  }
}

class _ServiceBanner extends StatelessWidget {
  const _ServiceBanner({required this.serviceUrl, required this.onRetry});

  final String serviceUrl;
  final VoidCallback onRetry;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Semantics(
      container: true,
      liveRegion: true,
      child: Material(
        color: scheme.errorContainer,
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
          child: Wrap(
            spacing: 16,
            runSpacing: 8,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: <Widget>[
              Text(
                'The local search service is not running at $serviceUrl. '
                'Start the backend, then select Retry.',
                style: TextStyle(color: scheme.onErrorContainer),
              ),
              FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
