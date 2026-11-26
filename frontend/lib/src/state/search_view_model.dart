import 'package:flutter/foundation.dart';

import '../api/models.dart';
import '../api/retrieval_api.dart';

/// Search query, filters and results for the search screen.
class SearchViewModel extends ChangeNotifier {
  SearchViewModel(this._api);

  final RetrievalApi _api;

  Set<SearchMode> _modes = SearchMode.values.toSet();
  Set<FileTypeFilter> _fileTypes = <FileTypeFilter>{};
  String _query = '';
  bool _loading = false;
  ApiException? _error;
  SearchResponse? _response;
  String _statusMessage = '';
  int _requestId = 0;

  Set<SearchMode> get modes => Set<SearchMode>.unmodifiable(_modes);
  Set<FileTypeFilter> get fileTypes =>
      Set<FileTypeFilter>.unmodifiable(_fileTypes);
  String get query => _query;
  bool get loading => _loading;
  ApiException? get error => _error;
  SearchResponse? get response => _response;
  List<SearchResult> get results =>
      _response?.results ?? const <SearchResult>[];

  /// Short text describing the last outcome, announced to screen readers.
  String get statusMessage => _statusMessage;

  bool isModeEnabled(SearchMode mode) => _modes.contains(mode);

  /// Turns a search method on or off. At least one method stays enabled.
  void toggleMode(SearchMode mode) {
    if (_modes.contains(mode)) {
      if (_modes.length == 1) {
        _statusMessage = 'At least one search method must stay on.';
        notifyListeners();
        return;
      }
      _modes = <SearchMode>{..._modes}..remove(mode);
    } else {
      _modes = <SearchMode>{..._modes, mode};
    }
    notifyListeners();
  }

  void toggleFileType(FileTypeFilter type) {
    _fileTypes = _fileTypes.contains(type)
        ? (<FileTypeFilter>{..._fileTypes}..remove(type))
        : <FileTypeFilter>{..._fileTypes, type};
    notifyListeners();
  }

  Future<void> search(String text, {required int topK}) async {
    final trimmed = text.trim();

    if (trimmed.isEmpty) {
      _statusMessage = 'Type something to search for.';
      notifyListeners();
      return;
    }

    final requestId = ++_requestId;
    _query = trimmed;
    _loading = true;
    _error = null;
    _statusMessage = 'Searching for "$trimmed"…';
    notifyListeners();

    final extensions = <String>[
      for (final type in _fileTypes) ...type.extensions,
    ];

    try {
      final response = await _api.search(
        SearchRequest(
          query: trimmed,
          topK: topK,
          modes: _modes,
          extensions: extensions.isEmpty ? null : extensions,
        ),
      );

      if (requestId != _requestId) {
        return; // A newer search started; ignore this stale response.
      }

      _response = response;
      _statusMessage = _describe(response);
    } on ApiException catch (error) {
      if (requestId != _requestId) {
        return;
      }
      _error = error;
      _response = null;
      _statusMessage = error.userMessage;
    } finally {
      if (requestId == _requestId) {
        _loading = false;
        notifyListeners();
      }
    }
  }

  void clear() {
    _requestId++;
    _query = '';
    _response = null;
    _error = null;
    _loading = false;
    _statusMessage = 'Search cleared.';
    notifyListeners();
  }

  static String _describe(SearchResponse response) {
    final count = response.results.length;
    final noun = count == 1 ? 'result' : 'results';
    final buffer = StringBuffer(
      count == 0
          ? 'No results for "${response.query}".'
          : '$count $noun for "${response.query}".',
    );

    if (response.warnings.isNotEmpty) {
      buffer.write(' Some search methods are unavailable.');
    }

    return buffer.toString();
  }
}
