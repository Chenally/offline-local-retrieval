/// Typed models for the local retrieval service's JSON API.
library;

enum SearchMode {
  keyword('keyword', 'Keywords'),
  textSemantic('text_semantic', 'Text meaning'),
  imageSemantic('image_semantic', 'Image content');

  const SearchMode(this.apiValue, this.label);

  final String apiValue;
  final String label;

  static SearchMode? fromApi(String value) {
    for (final mode in values) {
      if (mode.apiValue == value) {
        return mode;
      }
    }
    return null;
  }
}

enum FileTypeFilter {
  pdf('PDF', <String>['pdf']),
  word('Word', <String>['docx']),
  text('Text', <String>['txt']),
  images('Images', <String>['jpg', 'jpeg', 'png']);

  const FileTypeFilter(this.label, this.extensions);

  final String label;
  final List<String> extensions;
}

double _toDouble(Object? value) => value is num ? value.toDouble() : 0.0;

int _toInt(Object? value) => value is num ? value.toInt() : 0;

class ChannelScore {
  const ChannelScore({
    required this.raw,
    required this.normalized,
    required this.weight,
    required this.contribution,
    required this.rank,
  });

  factory ChannelScore.fromJson(Map<String, dynamic> json) {
    return ChannelScore(
      raw: _toDouble(json['raw']),
      normalized: _toDouble(json['normalized']),
      weight: _toDouble(json['weight']),
      contribution: _toDouble(json['contribution']),
      rank: _toInt(json['rank']),
    );
  }

  final double raw;
  final double normalized;
  final double weight;
  final double contribution;
  final int rank;
}

class SearchResult {
  const SearchResult({
    required this.documentId,
    required this.kind,
    required this.fileName,
    required this.extension,
    required this.paths,
    required this.score,
    required this.channelScores,
    this.snippet,
    this.pageNumber,
    this.chunkId,
  });

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    final scores = <SearchMode, ChannelScore>{};
    final rawScores = json['channel_scores'];

    if (rawScores is Map<String, dynamic>) {
      rawScores.forEach((key, value) {
        final mode = SearchMode.fromApi(key);
        if (mode != null && value is Map<String, dynamic>) {
          scores[mode] = ChannelScore.fromJson(value);
        }
      });
    }

    return SearchResult(
      documentId: json['document_id'] as String? ?? '',
      kind: json['kind'] as String? ?? 'text',
      fileName: json['file_name'] as String? ?? '',
      extension: json['extension'] as String? ?? '',
      paths: (json['paths'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<String>()
          .toList(),
      score: _toDouble(json['score']),
      channelScores: scores,
      snippet: json['snippet'] as String?,
      pageNumber: json['page_number'] as int?,
      chunkId: json['chunk_id'] as String?,
    );
  }

  final String documentId;
  final String kind;
  final String fileName;
  final String extension;
  final List<String> paths;
  final double score;
  final Map<SearchMode, ChannelScore> channelScores;
  final String? snippet;
  final int? pageNumber;
  final String? chunkId;

  bool get isImage => kind == 'image';

  String? get primaryPath => paths.isEmpty ? null : paths.first;
}

class SearchResponse {
  const SearchResponse({
    required this.query,
    required this.tookMs,
    required this.results,
    required this.warnings,
  });

  factory SearchResponse.fromJson(Map<String, dynamic> json) {
    return SearchResponse(
      query: json['query'] as String? ?? '',
      tookMs: _toDouble(json['took_ms']),
      results: (json['results'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(SearchResult.fromJson)
          .toList(),
      warnings: (json['warnings'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<String>()
          .toList(),
    );
  }

  final String query;
  final double tookMs;
  final List<SearchResult> results;
  final List<String> warnings;
}

class SearchRequest {
  const SearchRequest({
    required this.query,
    this.topK = 10,
    this.modes,
    this.extensions,
    this.pathPrefix,
  });

  final String query;
  final int topK;
  final Set<SearchMode>? modes;
  final List<String>? extensions;
  final String? pathPrefix;

  Map<String, dynamic> toJson() {
    return <String, dynamic>{
      'query': query,
      'top_k': topK,
      if (modes != null)
        'modes': modes!.map((mode) => mode.apiValue).toList(),
      if (extensions != null && extensions!.isNotEmpty) 'extensions': extensions,
      if (pathPrefix != null) 'path_prefix': pathPrefix,
    };
  }
}

class IndexError {
  const IndexError({
    required this.path,
    required this.errorType,
    required this.message,
  });

  factory IndexError.fromJson(Map<String, dynamic> json) {
    return IndexError(
      path: json['path'] as String? ?? '',
      errorType: json['error_type'] as String? ?? '',
      message: json['message'] as String? ?? '',
    );
  }

  final String path;
  final String errorType;
  final String message;
}

class IndexReport {
  const IndexReport({
    required this.root,
    required this.scanned,
    required this.indexed,
    required this.unchanged,
    required this.duplicates,
    required this.removed,
    required this.skippedUnsupported,
    required this.vectorCount,
    required this.errors,
  });

  factory IndexReport.fromJson(Map<String, dynamic> json) {
    return IndexReport(
      root: json['root'] as String? ?? '',
      scanned: _toInt(json['scanned']),
      indexed: _toInt(json['indexed']),
      unchanged: _toInt(json['unchanged']),
      duplicates: _toInt(json['duplicates']),
      removed: _toInt(json['removed']),
      skippedUnsupported: _toInt(json['skipped_unsupported']),
      vectorCount: _toInt(json['vector_count']),
      errors: (json['errors'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<Map<String, dynamic>>()
          .map(IndexError.fromJson)
          .toList(),
    );
  }

  final String root;
  final int scanned;
  final int indexed;
  final int unchanged;
  final int duplicates;
  final int removed;
  final int skippedUnsupported;
  final int vectorCount;
  final List<IndexError> errors;
}

enum JobState { queued, running, succeeded, failed }

class IndexJob {
  const IndexJob({
    required this.jobId,
    required this.operation,
    required this.directory,
    required this.state,
    required this.processed,
    required this.total,
    this.currentPath,
    this.report,
    this.removedCount = 0,
    this.error,
  });

  factory IndexJob.fromJson(Map<String, dynamic> json) {
    final stateName = json['state'] as String? ?? 'queued';
    final report = json['report'];

    return IndexJob(
      jobId: json['job_id'] as String? ?? '',
      operation: json['operation'] as String? ?? 'index',
      directory: json['directory'] as String? ?? '',
      state: JobState.values.firstWhere(
        (state) => state.name == stateName,
        orElse: () => JobState.queued,
      ),
      processed: _toInt(json['processed']),
      total: _toInt(json['total']),
      currentPath: json['current_path'] as String?,
      report: report is Map<String, dynamic>
          ? IndexReport.fromJson(report)
          : null,
      removedCount: _toInt(json['removed_count']),
      error: json['error'] as String?,
    );
  }

  final String jobId;
  final String operation;
  final String directory;
  final JobState state;
  final int processed;
  final int total;
  final String? currentPath;
  final IndexReport? report;
  final int removedCount;
  final String? error;

  bool get isFinished =>
      state == JobState.succeeded || state == JobState.failed;

  bool get isRemoval => operation == 'remove';

  /// Fraction complete, or null while the total is still unknown.
  double? get progress => total > 0 ? processed / total : null;
}

class IndexStatus {
  const IndexStatus({
    required this.roots,
    required this.fileCount,
    required this.documentCount,
    required this.textFileCount,
    required this.imageFileCount,
    required this.textVectorCount,
    required this.imageVectorCount,
    this.activeJob,
  });

  factory IndexStatus.fromJson(Map<String, dynamic> json) {
    final active = json['active_job'];

    return IndexStatus(
      roots: (json['roots'] as List<dynamic>? ?? const <dynamic>[])
          .whereType<String>()
          .toList(),
      fileCount: _toInt(json['file_count']),
      documentCount: _toInt(json['document_count']),
      textFileCount: _toInt(json['text_file_count']),
      imageFileCount: _toInt(json['image_file_count']),
      textVectorCount: _toInt(json['text_vector_count']),
      imageVectorCount: _toInt(json['image_vector_count']),
      activeJob: active is Map<String, dynamic>
          ? IndexJob.fromJson(active)
          : null,
    );
  }

  final List<String> roots;
  final int fileCount;
  final int documentCount;
  final int textFileCount;
  final int imageFileCount;
  final int textVectorCount;
  final int imageVectorCount;
  final IndexJob? activeJob;
}

class Readiness {
  const Readiness({required this.status, required this.components});

  factory Readiness.fromJson(Map<String, dynamic> json) {
    final components = <String, String>{};
    final raw = json['components'];

    if (raw is Map<String, dynamic>) {
      raw.forEach((key, value) => components[key] = '$value');
    }

    return Readiness(
      status: json['status'] as String? ?? 'unknown',
      components: components,
    );
  }

  final String status;
  final Map<String, String> components;

  bool get isReady => status == 'ready';
}
