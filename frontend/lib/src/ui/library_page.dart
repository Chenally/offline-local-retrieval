import 'package:material_ui/material_ui.dart';

import '../api/models.dart';
import '../platform.dart';
import '../state/library_view_model.dart';
import 'widgets/accessibility.dart';

class LibraryPage extends StatelessWidget {
  const LibraryPage({
    super.key,
    required this.viewModel,
    required this.platform,
  });

  final LibraryViewModel viewModel;
  final PlatformActions platform;

  Future<void> _addFolder(BuildContext context) async {
    final directory = await platform.pickDirectory();
    if (directory == null) {
      if (context.mounted) {
        announce(context, 'No folder selected.');
      }
      return;
    }
    await viewModel.addFolder(directory);
  }

  Future<void> _confirmRemove(BuildContext context, String folder) async {
    final confirmed = await showDialog<bool>(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Remove folder from index?'),
        content: Text(
          '$folder will no longer appear in search results. '
          'Your files are not deleted.',
        ),
        actions: <Widget>[
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(false),
            child: const Text('Cancel'),
          ),
          FilledButton(
            onPressed: () => Navigator.of(dialogContext).pop(true),
            child: const Text('Remove'),
          ),
        ],
      ),
    );

    if (confirmed == true) {
      await viewModel.removeFolder(folder);
    }
  }

  @override
  Widget build(BuildContext context) {
    return ListenableBuilder(
      listenable: viewModel,
      builder: (context, _) {
        final status = viewModel.status;
        final job = viewModel.activeJob;
        final finished = viewModel.lastFinishedJob;

        return ListView(
          padding: const EdgeInsets.all(24),
          children: <Widget>[
            const SectionHeading('Library', large: true),
            const SizedBox(height: 8),
            Text(
              'Choose folders to make searchable. Files are processed on this '
              'computer and never uploaded.',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: <Widget>[
                FilledButton.icon(
                  onPressed: viewModel.isBusy ? null : () => _addFolder(context),
                  icon: const Icon(Icons.create_new_folder_outlined),
                  label: const Text('Add folder'),
                ),
                OutlinedButton.icon(
                  onPressed: viewModel.loading ? null : viewModel.refresh,
                  icon: const Icon(Icons.refresh),
                  label: const Text('Refresh status'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            AnnouncedStatus(
              viewModel.statusMessage,
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            if (viewModel.error != null) ...<Widget>[
              const SizedBox(height: 16),
              ErrorPanel(
                message: viewModel.error!.userMessage,
                onRetry: viewModel.refresh,
              ),
            ],
            if (job != null) ...<Widget>[
              const SizedBox(height: 16),
              _JobProgress(job: job),
            ],
            if (finished?.report != null) ...<Widget>[
              const SizedBox(height: 16),
              _ReportSummary(report: finished!.report!),
            ],
            const SizedBox(height: 24),
            const SectionHeading('Index overview'),
            const SizedBox(height: 8),
            if (status != null) _Statistics(status: status) else const Text('Loading…'),
            const SizedBox(height: 24),
            const SectionHeading('Indexed folders'),
            const SizedBox(height: 8),
            if (viewModel.folders.isEmpty)
              const Text('No folders yet. Use "Add folder" to get started.')
            else
              FocusTraversalGroup(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: <Widget>[
                    for (final folder in viewModel.folders)
                      _FolderTile(
                        folder: folder,
                        enabled: !viewModel.isBusy,
                        onReindex: () => viewModel.reindex(folder),
                        onRemove: () => _confirmRemove(context, folder),
                      ),
                  ],
                ),
              ),
          ],
        );
      },
    );
  }
}

class _JobProgress extends StatelessWidget {
  const _JobProgress({required this.job});

  final IndexJob job;

  @override
  Widget build(BuildContext context) {
    final textTheme = Theme.of(context).textTheme;
    final value = job.progress;
    final progressText = job.total > 0
        ? '${job.processed} of ${job.total} files'
        : 'Preparing…';

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            SectionHeading(job.isRemoval ? 'Removing folder' : 'Indexing in progress'),
            const SizedBox(height: 8),
            Text(job.directory, style: textTheme.bodyMedium),
            const SizedBox(height: 12),
            LinearProgressIndicator(
              value: value,
              semanticsLabel: job.isRemoval ? 'Removal progress' : 'Indexing progress',
              semanticsValue: progressText,
            ),
            const SizedBox(height: 8),
            Text(progressText, style: textTheme.bodyMedium),
            if (job.currentPath != null)
              Text(
                'Current file: ${job.currentPath}',
                maxLines: 2,
                overflow: TextOverflow.ellipsis,
                style: textTheme.bodySmall,
              ),
          ],
        ),
      ),
    );
  }
}

class _ReportSummary extends StatelessWidget {
  const _ReportSummary({required this.report});

  final IndexReport report;

  @override
  Widget build(BuildContext context) {
    final rows = <MapEntry<String, int>>[
      MapEntry<String, int>('Files found', report.scanned),
      MapEntry<String, int>('New or changed', report.indexed),
      MapEntry<String, int>('Unchanged', report.unchanged),
      MapEntry<String, int>('Duplicates', report.duplicates),
      MapEntry<String, int>('Removed', report.removed),
      MapEntry<String, int>('Unsupported, skipped', report.skippedUnsupported),
      MapEntry<String, int>('Could not be read', report.errors.length),
    ];

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const SectionHeading('Last indexing result'),
            const SizedBox(height: 8),
            Wrap(
              spacing: 24,
              runSpacing: 8,
              children: <Widget>[
                for (final row in rows) Text('${row.key}: ${row.value}'),
              ],
            ),
            if (report.errors.isNotEmpty)
              ExpansionTile(
                tilePadding: EdgeInsets.zero,
                title: Text('Show ${report.errors.length} files that could not be read'),
                children: <Widget>[
                  for (final error in report.errors)
                    ListTile(
                      contentPadding: EdgeInsets.zero,
                      leading: const Icon(Icons.warning_amber_outlined),
                      title: Text(error.path),
                      subtitle: Text(error.message),
                    ),
                ],
              ),
          ],
        ),
      ),
    );
  }
}

class _Statistics extends StatelessWidget {
  const _Statistics({required this.status});

  final IndexStatus status;

  @override
  Widget build(BuildContext context) {
    final stats = <MapEntry<String, int>>[
      MapEntry<String, int>('Files', status.fileCount),
      MapEntry<String, int>('Text documents', status.textFileCount),
      MapEntry<String, int>('Images', status.imageFileCount),
      MapEntry<String, int>('Text passages', status.textVectorCount),
    ];

    return Wrap(
      spacing: 12,
      runSpacing: 12,
      children: <Widget>[
        for (final stat in stats)
          Semantics(
            container: true,
            label: '${stat.key}: ${stat.value}',
            excludeSemantics: true,
            child: Card(
              margin: EdgeInsets.zero,
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: <Widget>[
                    Text(
                      '${stat.value}',
                      style: Theme.of(context).textTheme.headlineSmall,
                    ),
                    Text(stat.key),
                  ],
                ),
              ),
            ),
          ),
      ],
    );
  }
}

class _FolderTile extends StatelessWidget {
  const _FolderTile({
    required this.folder,
    required this.enabled,
    required this.onReindex,
    required this.onRemove,
  });

  final String folder;
  final bool enabled;
  final VoidCallback onReindex;
  final VoidCallback onRemove;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        child: Row(
          children: <Widget>[
            const Icon(Icons.folder_outlined),
            const SizedBox(width: 12),
            // Long paths wrap instead of pushing the buttons off screen.
            Expanded(child: Text(folder)),
            const SizedBox(width: 8),
            IconButton(
              tooltip: 'Re-index $folder',
              icon: const Icon(Icons.sync),
              onPressed: enabled ? onReindex : null,
            ),
            IconButton(
              tooltip: 'Remove $folder from the index',
              icon: const Icon(Icons.delete_outline),
              onPressed: enabled ? onRemove : null,
            ),
          ],
        ),
      ),
    );
  }
}
