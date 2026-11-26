import 'package:frontend/src/api/models.dart';
import 'package:frontend/src/api/retrieval_api.dart';
import 'package:frontend/src/platform.dart';

SearchResult sampleResult({
  String fileName = 'budget.pdf',
  String kind = 'text',
  String extension = '.pdf',
  double score = 0.87,
  int? page = 2,
  String? snippet = 'The project budget covers model conversion.',
}) {
  return SearchResult(
    documentId: 'doc-$fileName',
    kind: kind,
    fileName: fileName,
    extension: extension,
    paths: <String>['/library/$fileName'],
    score: score,
    channelScores: const <SearchMode, ChannelScore>{
      SearchMode.keyword: ChannelScore(
        raw: 4.2,
        normalized: 1.0,
        weight: 0.3,
        contribution: 0.3,
        rank: 1,
      ),
      SearchMode.textSemantic: ChannelScore(
        raw: 0.61,
        normalized: 0.8,
        weight: 0.7,
        contribution: 0.56,
        rank: 2,
      ),
    },
    snippet: snippet,
    pageNumber: page,
  );
}

IndexJob sampleJob({
  JobState state = JobState.succeeded,
  int processed = 3,
  int total = 3,
  String operation = 'index',
  IndexReport? report,
}) {
  return IndexJob(
    jobId: 'job-1',
    operation: operation,
    directory: '/library',
    state: state,
    processed: processed,
    total: total,
    report: report ??
        (state == JobState.succeeded && operation == 'index'
            ? const IndexReport(
                root: '/library',
                scanned: 3,
                indexed: 2,
                unchanged: 1,
                duplicates: 0,
                removed: 0,
                skippedUnsupported: 0,
                vectorCount: 5,
                errors: <IndexError>[],
              )
            : null),
    removedCount: operation == 'remove' ? 3 : 0,
  );
}

IndexStatus sampleStatus({List<String> roots = const <String>['/library']}) {
  return IndexStatus(
    roots: roots,
    fileCount: 3,
    documentCount: 3,
    textFileCount: 2,
    imageFileCount: 1,
    textVectorCount: 5,
    imageVectorCount: 1,
  );
}

/// Scriptable in-memory implementation of [RetrievalApi].
class FakeRetrievalApi implements RetrievalApi {
  FakeRetrievalApi({
    this.alive = true,
    List<SearchResult>? results,
    IndexStatus? status,
  })  : results = results ?? <SearchResult>[sampleResult()],
        status = status ?? sampleStatus();

  @override
  Uri baseUri = Uri.parse('http://127.0.0.1:8765');

  bool alive;
  List<SearchResult> results;
  IndexStatus status;
  ApiException? searchError;
  ApiException? statusError;
  List<String> warnings = <String>[];

  /// Job states returned by successive [job] calls.
  List<IndexJob> jobSequence = <IndexJob>[sampleJob()];
  int _jobCalls = 0;

  final List<SearchRequest> searchRequests = <SearchRequest>[];
  final List<String> indexedDirectories = <String>[];
  final List<String> removedDirectories = <String>[];

  @override
  Future<bool> isAlive() async => alive;

  @override
  Future<Readiness> readiness() async {
    if (!alive) {
      throw const ApiException(ApiErrorKind.unreachable, 'offline');
    }
    return const Readiness(
      status: 'ready',
      components: <String, String>{'text_semantic': 'ready', 'mobileclip': 'ready'},
    );
  }

  @override
  Future<IndexJob> startIndexing(String directory) async {
    indexedDirectories.add(directory);
    _jobCalls = 0;
    return sampleJob(state: JobState.running, processed: 0, total: 3);
  }

  @override
  Future<IndexJob> removeFolder(String directory) async {
    removedDirectories.add(directory);
    _jobCalls = 0;
    jobSequence = <IndexJob>[sampleJob(operation: 'remove')];
    return sampleJob(state: JobState.running, operation: 'remove', processed: 0, total: 0);
  }

  @override
  Future<IndexJob> job(String jobId) async {
    final index = _jobCalls < jobSequence.length ? _jobCalls : jobSequence.length - 1;
    _jobCalls++;
    return jobSequence[index];
  }

  @override
  Future<IndexStatus> indexStatus() async {
    if (statusError != null) {
      throw statusError!;
    }
    return status;
  }

  @override
  Future<SearchResponse> search(SearchRequest request) async {
    searchRequests.add(request);
    if (searchError != null) {
      throw searchError!;
    }
    return SearchResponse(
      query: request.query,
      tookMs: 12,
      results: results,
      warnings: warnings,
    );
  }
}

class RecordingPlatform {
  final List<String> opened = <String>[];
  final List<String> revealed = <String>[];
  String? nextDirectory = '/new-folder';
  bool openSucceeds = true;

  PlatformActions get actions => PlatformActions(
        pickDirectory: () async => nextDirectory,
        openPath: (path) async {
          opened.add(path);
          return openSucceeds;
        },
        revealPath: (path) async {
          revealed.add(path);
          return openSucceeds;
        },
      );
}
