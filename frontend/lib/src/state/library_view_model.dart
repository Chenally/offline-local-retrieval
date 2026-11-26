import 'package:flutter/foundation.dart';

import '../api/models.dart';
import '../api/retrieval_api.dart';

/// Indexed folders, index statistics and indexing-job progress.
class LibraryViewModel extends ChangeNotifier {
  LibraryViewModel(
    this._api, {
    this.pollInterval = const Duration(milliseconds: 700),
  });

  final RetrievalApi _api;
  final Duration pollInterval;

  IndexStatus? _status;
  IndexJob? _activeJob;
  IndexJob? _lastFinishedJob;
  bool _loading = false;
  ApiException? _error;
  String _statusMessage = '';
  bool _disposed = false;
  int _lastAnnouncedQuarter = 0;

  IndexStatus? get status => _status;
  IndexJob? get activeJob => _activeJob;
  IndexJob? get lastFinishedJob => _lastFinishedJob;
  bool get loading => _loading;
  ApiException? get error => _error;
  bool get isBusy => _activeJob != null;
  List<String> get folders => _status?.roots ?? const <String>[];

  /// Short text describing progress or the last outcome, for announcements.
  String get statusMessage => _statusMessage;

  Future<void> refresh() async {
    _loading = true;
    notifyListeners();

    try {
      _status = await _api.indexStatus();
      _error = null;

      final serverJob = _status?.activeJob;
      if (serverJob != null && _activeJob == null && !serverJob.isFinished) {
        // A job started elsewhere (or before a restart): follow it.
        _activeJob = serverJob;
        _loading = false;
        _notify();
        await _follow(serverJob);
        return;
      }
    } on ApiException catch (error) {
      _error = error;
      _statusMessage = error.userMessage;
    } finally {
      _loading = false;
      _notify();
    }
  }

  Future<void> addFolder(String directory) =>
      _run(() => _api.startIndexing(directory));

  Future<void> reindex(String directory) =>
      _run(() => _api.startIndexing(directory));

  Future<void> removeFolder(String directory) =>
      _run(() => _api.removeFolder(directory));

  Future<void> _run(Future<IndexJob> Function() start) async {
    if (isBusy) {
      _statusMessage = 'Please wait for the current indexing job to finish.';
      _notify();
      return;
    }

    try {
      final job = await start();
      _activeJob = job;
      _error = null;
      _lastAnnouncedQuarter = 0;
      _statusMessage = job.isRemoval
          ? 'Removing ${_folderName(job.directory)} from the index…'
          : 'Indexing ${_folderName(job.directory)}…';
      _notify();
      await _follow(job);
    } on ApiException catch (error) {
      _error = error;
      _statusMessage = error.userMessage;
      _notify();
    }
  }

  Future<void> _follow(IndexJob initial) async {
    var job = initial;

    try {
      while (!job.isFinished && !_disposed) {
        await Future<void>.delayed(pollInterval);
        if (_disposed) {
          return;
        }
        job = await _api.job(job.jobId);
        _activeJob = job;
        _announceMilestone(job);
        _notify();
      }
    } on ApiException catch (error) {
      _error = error;
      _statusMessage = error.userMessage;
      _activeJob = null;
      _notify();
      return;
    }

    if (_disposed) {
      return;
    }

    _activeJob = null;
    _lastFinishedJob = job;
    _statusMessage = describeFinishedJob(job);
    _notify();

    try {
      _status = await _api.indexStatus();
    } on ApiException catch (error) {
      _error = error;
    }
    _notify();
  }

  void _announceMilestone(IndexJob job) {
    final progress = job.progress;
    if (progress == null || job.isRemoval) {
      return;
    }

    final quarter = (progress * 4).floor();
    if (quarter > _lastAnnouncedQuarter && quarter < 4) {
      _lastAnnouncedQuarter = quarter;
      _statusMessage =
          'Indexing ${quarter * 25} percent: ${job.processed} of ${job.total} files.';
    }
  }

  void _notify() {
    if (!_disposed) {
      notifyListeners();
    }
  }

  @override
  void dispose() {
    _disposed = true;
    super.dispose();
  }

  static String describeFinishedJob(IndexJob job) {
    final name = _folderName(job.directory);

    if (job.state == JobState.failed) {
      return 'Indexing $name failed: ${job.error ?? 'unknown error'}.';
    }

    if (job.isRemoval) {
      final files = job.removedCount == 1 ? 'file' : 'files';
      return 'Removed $name from the index (${job.removedCount} $files).';
    }

    final report = job.report;
    if (report == null) {
      return 'Finished indexing $name.';
    }

    final parts = <String>[
      '${report.indexed} new or changed',
      '${report.unchanged} unchanged',
      if (report.duplicates > 0) '${report.duplicates} duplicates',
      if (report.removed > 0) '${report.removed} removed',
      if (report.errors.isNotEmpty) '${report.errors.length} could not be read',
    ];

    return 'Finished indexing $name: ${parts.join(', ')}.';
  }

  static String _folderName(String path) {
    final cleaned = path.replaceAll(RegExp(r'[\\/]+$'), '');
    final segments = cleaned.split(RegExp(r'[\\/]'));
    return segments.isEmpty || segments.last.isEmpty ? path : segments.last;
  }
}
