import 'dart:io';

import 'package:material_ui/material_ui.dart';

import '../../api/models.dart';
import 'accessibility.dart';

/// One search result. The whole card is a single, keyboard-focusable button
/// that opens the file; its screen-reader label summarizes the result and
/// explains why it matched.
class ResultCard extends StatelessWidget {
  const ResultCard({
    super.key,
    required this.result,
    required this.position,
    required this.total,
    required this.onOpen,
    required this.onShowInFolder,
  });

  final SearchResult result;
  final int position;
  final int total;
  final VoidCallback onOpen;
  final VoidCallback onShowInFolder;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return FocusOutline(
      child: Card(
        margin: EdgeInsets.zero,
        clipBehavior: Clip.antiAlias,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Expanded(
              child: Semantics(
                button: true,
                label: semanticLabel(),
                hint: 'Opens the file',
                onTap: onOpen,
                excludeSemantics: true,
                child: InkWell(
                  onTap: onOpen,
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: <Widget>[
                        _Thumbnail(result: result),
                        const SizedBox(width: 16),
                        Expanded(child: _Details(result: result, theme: theme)),
                      ],
                    ),
                  ),
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.all(8),
              child: IconButton(
                tooltip: 'Show ${result.fileName} in its folder',
                icon: const Icon(Icons.folder_open),
                onPressed: result.primaryPath == null ? null : onShowInFolder,
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Everything a screen-reader user needs in one utterance.
  String semanticLabel() {
    final parts = <String>[
      'Result $position of $total',
      result.fileName,
      result.isImage ? 'image' : '${_typeName(result.extension)} document',
      if (result.pageNumber != null) 'page ${result.pageNumber}',
      'relevance ${(result.score * 100).round()} percent',
      'matched by ${matchExplanation()}',
      if (result.paths.length > 1) '${result.paths.length} copies',
      if (result.snippet != null && result.snippet!.trim().isNotEmpty)
        'excerpt: ${_shorten(result.snippet!, 160)}',
    ];
    return parts.join(', ');
  }

  String matchExplanation() {
    final entries = result.channelScores.entries.toList()
      ..sort((a, b) => b.value.contribution.compareTo(a.value.contribution));

    if (entries.isEmpty) {
      return 'no channel';
    }

    return entries
        .map((entry) => '${entry.key.label.toLowerCase()} '
            '${(entry.value.normalized * 100).round()} percent')
        .join(' and ');
  }

  static String _typeName(String extension) {
    switch (extension.toLowerCase()) {
      case '.pdf':
        return 'PDF';
      case '.docx':
        return 'Word';
      case '.txt':
        return 'text';
      default:
        return extension.replaceFirst('.', '').toUpperCase();
    }
  }

  static String _shorten(String text, int maxLength) {
    final collapsed = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    return collapsed.length <= maxLength
        ? collapsed
        : '${collapsed.substring(0, maxLength)}…';
  }
}

class _Thumbnail extends StatelessWidget {
  const _Thumbnail({required this.result});

  final SearchResult result;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    final path = result.primaryPath;
    final fallback = Icon(
      _iconFor(result),
      size: 40,
      color: scheme.primary,
    );

    if (!result.isImage || path == null) {
      return SizedBox(width: 64, height: 64, child: Center(child: fallback));
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(8),
      child: Image.file(
        File(path),
        width: 64,
        height: 64,
        fit: BoxFit.cover,
        cacheWidth: 128,
        excludeFromSemantics: true,
        errorBuilder: (context, error, stackTrace) =>
            SizedBox(width: 64, height: 64, child: Center(child: fallback)),
      ),
    );
  }

  static IconData _iconFor(SearchResult result) {
    if (result.isImage) {
      return Icons.image_outlined;
    }
    switch (result.extension.toLowerCase()) {
      case '.pdf':
        return Icons.picture_as_pdf_outlined;
      case '.docx':
        return Icons.description_outlined;
      default:
        return Icons.article_outlined;
    }
  }
}

class _Details extends StatelessWidget {
  const _Details({required this.result, required this.theme});

  final SearchResult result;
  final ThemeData theme;

  @override
  Widget build(BuildContext context) {
    final textTheme = theme.textTheme;
    final subtle = theme.colorScheme.onSurfaceVariant;
    final details = <String>[
      if (result.pageNumber != null) 'Page ${result.pageNumber}',
      '${(result.score * 100).round()}% match',
      if (result.paths.length > 1) '${result.paths.length} copies',
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(result.fileName, style: textTheme.titleMedium),
        const SizedBox(height: 4),
        Text(details.join(' · '), style: textTheme.bodyMedium?.copyWith(color: subtle)),
        if (result.snippet != null && result.snippet!.trim().isNotEmpty) ...<Widget>[
          const SizedBox(height: 8),
          Text(
            result.snippet!.replaceAll(RegExp(r'\s+'), ' ').trim(),
            maxLines: 3,
            overflow: TextOverflow.ellipsis,
            style: textTheme.bodyMedium,
          ),
        ],
        if (result.primaryPath != null) ...<Widget>[
          const SizedBox(height: 8),
          Text(
            result.primaryPath!,
            maxLines: 2,
            overflow: TextOverflow.ellipsis,
            style: textTheme.bodySmall?.copyWith(color: subtle),
          ),
        ],
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 4,
          children: <Widget>[
            for (final entry in result.channelScores.entries)
              _MatchBadge(mode: entry.key, score: entry.value),
          ],
        ),
      ],
    );
  }
}

class _MatchBadge extends StatelessWidget {
  const _MatchBadge({required this.mode, required this.score});

  final SearchMode mode;
  final ChannelScore score;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return DecoratedBox(
      decoration: BoxDecoration(
        color: scheme.secondaryContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
        child: Text(
          '${mode.label} ${(score.normalized * 100).round()}%',
          style: Theme.of(context)
              .textTheme
              .labelMedium
              ?.copyWith(color: scheme.onSecondaryContainer),
        ),
      ),
    );
  }
}
