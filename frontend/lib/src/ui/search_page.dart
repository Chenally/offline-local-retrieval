import 'package:material_ui/material_ui.dart';
import 'package:flutter/services.dart';

import '../api/models.dart';
import '../platform.dart';
import '../state/search_view_model.dart';
import '../state/settings_controller.dart';
import 'widgets/accessibility.dart';
import 'widgets/result_card.dart';

class SearchPage extends StatefulWidget {
  const SearchPage({
    super.key,
    required this.viewModel,
    required this.settings,
    required this.platform,
    required this.searchFocusNode,
  });

  final SearchViewModel viewModel;
  final SettingsController settings;
  final PlatformActions platform;
  final FocusNode searchFocusNode;

  @override
  State<SearchPage> createState() => _SearchPageState();
}

class _SearchPageState extends State<SearchPage> {
  late final TextEditingController _controller = TextEditingController(
    text: widget.viewModel.query,
  );

  // The clear button stays available to mouse and screen-reader users but is
  // skipped by Tab: keyboard users clear with Escape, and Tab goes straight
  // from the search box to the Search button.
  final FocusNode _clearButtonFocus = FocusNode(skipTraversal: true);

  @override
  void dispose() {
    _controller.dispose();
    _clearButtonFocus.dispose();
    super.dispose();
  }

  void _search() {
    widget.viewModel.search(
      _controller.text,
      topK: widget.settings.value.resultsPerSearch,
    );
  }

  void _clear() {
    _controller.clear();
    widget.viewModel.clear();
    widget.searchFocusNode.requestFocus();
  }

  Future<void> _open(SearchResult result, {required bool reveal}) async {
    final path = result.primaryPath;
    if (path == null) {
      return;
    }

    final opened = reveal
        ? await widget.platform.revealPath(path)
        : await widget.platform.openPath(path);

    if (!opened && mounted) {
      final message = 'Could not open ${result.fileName}. '
          'The file may have been moved or deleted.';
      ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text(message)));
      announce(context, message);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: widget.viewModel,
      builder: (context, _) {
        final vm = widget.viewModel;
        final results = vm.results;

        return ListView(
          padding: const EdgeInsets.all(24),
          children: <Widget>[
            const SectionHeading('Search', large: true),
            const SizedBox(height: 16),
            _buildSearchField(vm),
            const SizedBox(height: 16),
            _buildFilters(vm),
            const SizedBox(height: 16),
            if (vm.loading)
              const LinearProgressIndicator(semanticsLabel: 'Searching'),
            AnnouncedStatus(
              vm.statusMessage,
              style: Theme.of(context).textTheme.bodyLarge,
              // The error panel below shows the same text.
              visible: vm.error == null,
            ),
            if (vm.response != null && vm.response!.warnings.isNotEmpty)
              for (final warning in vm.response!.warnings)
                Padding(
                  padding: const EdgeInsets.only(top: 8),
                  child: Text(
                    warning,
                    style: TextStyle(color: Theme.of(context).colorScheme.error),
                  ),
                ),
            if (vm.error != null) ...<Widget>[
              ErrorPanel(message: vm.error!.userMessage, onRetry: _search),
            ],
            const SizedBox(height: 16),
            if (results.isNotEmpty)
              Semantics(
                container: true,
                explicitChildNodes: true,
                label: 'Search results',
                child: FocusTraversalGroup(
                  child: Column(
                    children: <Widget>[
                      for (var i = 0; i < results.length; i++)
                        Padding(
                          padding: const EdgeInsets.only(bottom: 12),
                          child: ResultCard(
                            result: results[i],
                            position: i + 1,
                            total: results.length,
                            onOpen: () => _open(results[i], reveal: false),
                            onShowInFolder: () => _open(results[i], reveal: true),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            if (results.isEmpty && vm.response == null && vm.error == null && !vm.loading)
              const _SearchTips(),
          ],
        );
      },
    );
  }

  Widget _buildSearchField(SearchViewModel vm) {
    return CallbackShortcuts(
      bindings: <ShortcutActivator, VoidCallback>{
        const SingleActivator(LogicalKeyboardKey.escape): _clear,
      },
      child: Wrap(
        spacing: 12,
        runSpacing: 12,
        crossAxisAlignment: WrapCrossAlignment.start,
        children: <Widget>[
          ConstrainedBox(
            constraints: const BoxConstraints(minWidth: 240, maxWidth: 640),
            child: TextField(
              controller: _controller,
              focusNode: widget.searchFocusNode,
              autofocus: true,
              textInputAction: TextInputAction.search,
              onSubmitted: (_) => _search(),
              // Keep focus in the box after Enter so keyboard users can
              // refine the query or Tab onward to the results.
              onEditingComplete: () {},
              decoration: InputDecoration(
                labelText: 'Search your files',
                hintText: 'For example: budget report, or a red car',
                helperText: 'Press Enter to search, Escape to clear',
                prefixIcon: const Icon(Icons.search),
                suffixIcon: _controller.text.isEmpty
                    ? null
                    : IconButton(
                        focusNode: _clearButtonFocus,
                        tooltip: 'Clear search',
                        icon: const Icon(Icons.clear),
                        onPressed: _clear,
                      ),
              ),
              onChanged: (_) => setState(() {}),
            ),
          ),
          Padding(
            // Centres the 48 px button on the 56 px text box.
            padding: const EdgeInsets.only(top: 4),
            child: FilledButton.icon(
              onPressed: vm.loading ? null : _search,
              icon: const Icon(Icons.search),
              label: const Text('Search'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFilters(SearchViewModel vm) {
    final labelStyle = Theme.of(context).textTheme.labelLarge;

    return FocusTraversalGroup(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Semantics(header: true, child: Text('Search methods', style: labelStyle)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: <Widget>[
              for (final mode in SearchMode.values)
                FilterChip(
                  label: Text(mode.label),
                  tooltip: _modeDescription(mode),
                  selected: vm.isModeEnabled(mode),
                  onSelected: (_) => vm.toggleMode(mode),
                ),
            ],
          ),
          const SizedBox(height: 16),
          Semantics(header: true, child: Text('File types', style: labelStyle)),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: <Widget>[
              for (final type in FileTypeFilter.values)
                FilterChip(
                  label: Text(type.label),
                  selected: vm.fileTypes.contains(type),
                  onSelected: (_) => vm.toggleFileType(type),
                ),
            ],
          ),
          if (vm.fileTypes.isEmpty)
            Padding(
              padding: const EdgeInsets.only(top: 4),
              child: Text(
                'All file types are included.',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ),
        ],
      ),
    );
  }

  static String _modeDescription(SearchMode mode) {
    switch (mode) {
      case SearchMode.keyword:
        return 'Find files containing the exact words';
      case SearchMode.textSemantic:
        return 'Find documents with a similar meaning';
      case SearchMode.imageSemantic:
        return 'Find images that match a description';
    }
  }
}

class _SearchTips extends StatelessWidget {
  const _SearchTips();

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const SectionHeading('Tips'),
            const SizedBox(height: 8),
            Text(
              '• Describe what you remember, not only exact words.\n'
              '• Describe a photo to find images, for example "a dog on a beach".\n'
              '• Add folders on the Library page before searching.\n'
              '• Press Control+F (Command+F on Mac) to return to the search box.',
              style: textTheme.bodyMedium,
            ),
          ],
        ),
      ),
    );
  }
}
