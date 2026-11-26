import 'package:material_ui/material_ui.dart';

import '../api/models.dart';
import '../api/retrieval_api.dart';
import '../state/settings_controller.dart';
import 'widgets/accessibility.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({
    super.key,
    required this.settings,
    required this.api,
    required this.onServiceUrlChanged,
  });

  final SettingsController settings;
  final RetrievalApi api;
  final VoidCallback onServiceUrlChanged;

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  late final TextEditingController _urlController = TextEditingController(
    text: widget.settings.value.serviceUrl,
  );
  String? _urlError;
  String _connectionMessage = '';
  Readiness? _readiness;
  bool _checking = false;

  @override
  void dispose() {
    _urlController.dispose();
    super.dispose();
  }

  Future<void> _saveUrlAndTest() async {
    final text = _urlController.text.trim();
    final uri = Uri.tryParse(text);

    if (uri == null || !(uri.scheme == 'http' || uri.scheme == 'https') || uri.host.isEmpty) {
      setState(() {
        _urlError = 'Enter a full address such as $defaultServiceUrl';
        _connectionMessage = _urlError!;
      });
      return;
    }

    setState(() {
      _urlError = null;
      _checking = true;
      _connectionMessage = 'Checking connection…';
    });

    await widget.settings.setServiceUrl(text);
    widget.onServiceUrlChanged();

    try {
      final readiness = await widget.api.readiness();
      if (!mounted) {
        return;
      }
      setState(() {
        _readiness = readiness;
        _connectionMessage = readiness.isReady
            ? 'Connected. All search components are ready.'
            : 'Connected, but some components are unavailable.';
      });
    } on ApiException catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _readiness = null;
        _connectionMessage = error.userMessage;
      });
    } finally {
      if (mounted) {
        setState(() => _checking = false);
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.settings,
      builder: (context, _) {
        final value = widget.settings.value;
        final textTheme = Theme.of(context).textTheme;

        return ListView(
          padding: const EdgeInsets.all(24),
          children: <Widget>[
            const SectionHeading('Settings', large: true),
            const SizedBox(height: 24),
            const SectionHeading('Appearance'),
            const SizedBox(height: 12),
            Text('Theme', style: textTheme.labelLarge),
            const SizedBox(height: 8),
            SegmentedButton<ThemeMode>(
              segments: const <ButtonSegment<ThemeMode>>[
                ButtonSegment<ThemeMode>(
                  value: ThemeMode.system,
                  label: Text('System'),
                  icon: Icon(Icons.brightness_auto),
                ),
                ButtonSegment<ThemeMode>(
                  value: ThemeMode.light,
                  label: Text('Light'),
                  icon: Icon(Icons.light_mode),
                ),
                ButtonSegment<ThemeMode>(
                  value: ThemeMode.dark,
                  label: Text('Dark'),
                  icon: Icon(Icons.dark_mode),
                ),
              ],
              selected: <ThemeMode>{value.themeMode},
              onSelectionChanged: (selection) =>
                  widget.settings.setThemeMode(selection.first),
            ),
            const SizedBox(height: 12),
            SwitchListTile(
              contentPadding: EdgeInsets.zero,
              title: const Text('High contrast'),
              subtitle: const Text(
                'Uses black, white and bright accent colors with stronger outlines.',
              ),
              value: value.highContrast,
              onChanged: widget.settings.setHighContrast,
            ),
            const SizedBox(height: 12),
            Text(
              'Text size: ${(value.textScale * 100).round()}%',
              style: textTheme.labelLarge,
            ),
            Semantics(
              label: 'Text size',
              child: Slider(
              value: value.textScale,
              min: minTextScale,
              max: maxTextScale,
              divisions: 4,
              label: '${(value.textScale * 100).round()}%',
              semanticFormatterCallback: (scale) =>
                  'Text size ${(scale * 100).round()} percent',
              onChanged: widget.settings.setTextScale,
              ),
            ),
            Text(
              'Applies on top of your system text size. Layouts adapt up to 200%.',
              style: textTheme.bodySmall,
            ),
            const SizedBox(height: 24),
            const SectionHeading('Search'),
            const SizedBox(height: 12),
            Text('Results per search', style: textTheme.labelLarge),
            const SizedBox(height: 8),
            SegmentedButton<int>(
              segments: <ButtonSegment<int>>[
                for (final count in resultCountOptions)
                  ButtonSegment<int>(value: count, label: Text('$count')),
              ],
              selected: <int>{value.resultsPerSearch},
              onSelectionChanged: (selection) =>
                  widget.settings.setResultsPerSearch(selection.first),
            ),
            const SizedBox(height: 24),
            const SectionHeading('Local search service'),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: <Widget>[
                ConstrainedBox(
                  constraints: const BoxConstraints(minWidth: 240, maxWidth: 480),
                  child: TextField(
                    controller: _urlController,
                    keyboardType: TextInputType.url,
                    onSubmitted: (_) => _saveUrlAndTest(),
                    decoration: InputDecoration(
                      labelText: 'Service address',
                      helperText: 'Default: $defaultServiceUrl',
                      errorText: _urlError,
                    ),
                  ),
                ),
                FilledButton.icon(
                  onPressed: _checking ? null : _saveUrlAndTest,
                  icon: const Icon(Icons.wifi_tethering),
                  label: const Text('Save and test connection'),
                ),
              ],
            ),
            const SizedBox(height: 8),
            AnnouncedStatus(_connectionMessage, style: textTheme.bodyLarge),
            if (_readiness != null)
              for (final entry in _readiness!.components.entries)
                Text('${_componentName(entry.key)}: ${entry.value}'),
            const SizedBox(height: 24),
            const SectionHeading('Keyboard shortcuts'),
            const SizedBox(height: 8),
            const _ShortcutTable(),
          ],
        );
      },
    );
  }

  static String _componentName(String key) {
    switch (key) {
      case 'text_semantic':
        return 'Text meaning model';
      case 'mobileclip':
        return 'Image model';
      case 'vector_store':
        return 'Vector store';
      case 'keyword_index':
        return 'Keyword index';
      case 'manifest':
        return 'Index manifest';
      default:
        return key;
    }
  }
}

class _ShortcutTable extends StatelessWidget {
  const _ShortcutTable();

  static const List<MapEntry<String, String>> _shortcuts = <MapEntry<String, String>>[
    MapEntry<String, String>('Ctrl+F (Cmd+F on Mac)', 'Go to the search box'),
    MapEntry<String, String>('Ctrl+1 / Ctrl+2 / Ctrl+3', 'Open Search, Library or Settings'),
    MapEntry<String, String>('Enter', 'Run the search, or open the focused result'),
    MapEntry<String, String>('Escape', 'Clear the search box'),
    MapEntry<String, String>('Tab / Shift+Tab', 'Move to the next or previous control'),
    MapEntry<String, String>('Space', 'Toggle the focused filter or switch'),
  ];

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        for (final shortcut in _shortcuts)
          Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Text('${shortcut.key}: ${shortcut.value}'),
          ),
      ],
    );
  }
}
