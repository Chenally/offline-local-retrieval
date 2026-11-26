import 'package:material_ui/material_ui.dart';

import 'api/retrieval_api.dart';
import 'platform.dart';
import 'state/library_view_model.dart';
import 'state/search_view_model.dart';
import 'state/settings_controller.dart';
import 'ui/home_shell.dart';
import 'ui/theme.dart';

class OfflineRetrievalApp extends StatefulWidget {
  const OfflineRetrievalApp({
    super.key,
    required this.settings,
    required this.api,
    required this.platform,
    this.libraryPollInterval = const Duration(milliseconds: 700),
  });

  final SettingsController settings;
  final RetrievalApi api;
  final PlatformActions platform;
  final Duration libraryPollInterval;

  @override
  State<OfflineRetrievalApp> createState() => _OfflineRetrievalAppState();
}

class _OfflineRetrievalAppState extends State<OfflineRetrievalApp> {
  late final SearchViewModel _search = SearchViewModel(widget.api);
  late final LibraryViewModel _library = LibraryViewModel(
    widget.api,
    pollInterval: widget.libraryPollInterval,
  );

  @override
  void dispose() {
    _search.dispose();
    _library.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.settings,
      builder: (context, _) {
        final settings = widget.settings.value;
        final forceContrast = settings.highContrast;

        final light = buildTheme(brightness: Brightness.light, highContrast: forceContrast);
        final dark = buildTheme(brightness: Brightness.dark, highContrast: forceContrast);

        return MaterialApp(
          title: 'Offline Local Search',
          debugShowCheckedModeBanner: false,
          themeMode: settings.themeMode,
          theme: light,
          darkTheme: dark,
          // Follow the operating system's high-contrast setting as well.
          highContrastTheme: buildTheme(brightness: Brightness.light, highContrast: true),
          highContrastDarkTheme: buildTheme(brightness: Brightness.dark, highContrast: true),
          builder: (context, child) {
            final media = MediaQuery.of(context);
            final systemFactor = media.textScaler.scale(14) / 14;
            return MediaQuery(
              data: media.copyWith(
                textScaler: TextScaler.linear(systemFactor * settings.textScale),
              ),
              child: child ?? const SizedBox.shrink(),
            );
          },
          home: HomeShell(
            api: widget.api,
            settings: widget.settings,
            search: _search,
            library: _library,
            platform: widget.platform,
          ),
        );
      },
    );
  }
}
