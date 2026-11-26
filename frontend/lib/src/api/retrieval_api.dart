import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import 'models.dart';

enum ApiErrorKind { unreachable, badRequest, notFound, notReady, server }

/// A failure talking to the local retrieval service, with a message that can
/// be shown to (and read aloud for) the user.
class ApiException implements Exception {
  const ApiException(this.kind, this.message, {this.statusCode});

  final ApiErrorKind kind;
  final String message;
  final int? statusCode;

  String get userMessage {
    switch (kind) {
      case ApiErrorKind.unreachable:
        return 'The local search service is not responding. '
            'Make sure it is running, then try again.';
      case ApiErrorKind.notReady:
        return 'The search service is starting or its models are missing. '
            '$message';
      case ApiErrorKind.badRequest:
      case ApiErrorKind.notFound:
        return message;
      case ApiErrorKind.server:
        return 'The search service reported an error: $message';
    }
  }

  @override
  String toString() => 'ApiException($kind, $statusCode, $message)';
}

/// Operations the user interface needs from the local service.
abstract class RetrievalApi {
  Uri get baseUri;
  set baseUri(Uri value);

  Future<bool> isAlive();
  Future<Readiness> readiness();
  Future<IndexJob> startIndexing(String directory);
  Future<IndexJob> removeFolder(String directory);
  Future<IndexJob> job(String jobId);
  Future<IndexStatus> indexStatus();
  Future<SearchResponse> search(SearchRequest request);
}

class HttpRetrievalApi implements RetrievalApi {
  HttpRetrievalApi({
    required this.baseUri,
    http.Client? client,
    this.timeout = const Duration(seconds: 15),
  }) : _client = client ?? http.Client();

  final http.Client _client;
  final Duration timeout;

  @override
  Uri baseUri;

  @override
  Future<bool> isAlive() async {
    try {
      final body = await _send('GET', 'health/live');
      return body['status'] == 'ok';
    } on ApiException {
      return false;
    }
  }

  @override
  Future<Readiness> readiness() async {
    return Readiness.fromJson(await _send('GET', 'health/ready'));
  }

  @override
  Future<IndexJob> startIndexing(String directory) async {
    return IndexJob.fromJson(
      await _send('POST', 'index', body: <String, dynamic>{'directory': directory}),
    );
  }

  @override
  Future<IndexJob> removeFolder(String directory) async {
    return IndexJob.fromJson(
      await _send(
        'POST',
        'index/remove',
        body: <String, dynamic>{'directory': directory},
      ),
    );
  }

  @override
  Future<IndexJob> job(String jobId) async {
    return IndexJob.fromJson(
      await _send('GET', 'index/jobs/${Uri.encodeComponent(jobId)}'),
    );
  }

  @override
  Future<IndexStatus> indexStatus() async {
    return IndexStatus.fromJson(await _send('GET', 'index/status'));
  }

  @override
  Future<SearchResponse> search(SearchRequest request) async {
    return SearchResponse.fromJson(
      await _send('POST', 'search', body: request.toJson()),
    );
  }

  Uri _resolve(String path) {
    final base = baseUri.toString().endsWith('/')
        ? baseUri
        : Uri.parse('${baseUri.toString()}/');
    return base.resolve(path);
  }

  Future<Map<String, dynamic>> _send(
    String method,
    String path, {
    Map<String, dynamic>? body,
  }) async {
    final request = http.Request(method, _resolve(path));
    request.headers['Accept'] = 'application/json';

    if (body != null) {
      request.headers['Content-Type'] = 'application/json';
      request.body = jsonEncode(body);
    }

    final http.Response response;

    try {
      final streamed = await _client.send(request).timeout(timeout);
      response = await http.Response.fromStream(streamed).timeout(timeout);
    } on TimeoutException {
      throw const ApiException(
        ApiErrorKind.unreachable,
        'The request timed out',
      );
    } on http.ClientException catch (error) {
      throw ApiException(ApiErrorKind.unreachable, error.message);
    } catch (error) {
      throw ApiException(ApiErrorKind.unreachable, '$error');
    }

    final Object? decoded = _decode(response.body);

    if (response.statusCode >= 200 && response.statusCode < 300) {
      if (decoded is Map<String, dynamic>) {
        return decoded;
      }
      throw ApiException(
        ApiErrorKind.server,
        'Unexpected response format',
        statusCode: response.statusCode,
      );
    }

    final message = _detail(decoded) ?? response.reasonPhrase ?? 'Unknown error';

    throw ApiException(
      _kindFor(response.statusCode),
      message,
      statusCode: response.statusCode,
    );
  }

  static Object? _decode(String body) {
    if (body.isEmpty) {
      return <String, dynamic>{};
    }
    try {
      return jsonDecode(body);
    } on FormatException {
      return null;
    }
  }

  static String? _detail(Object? decoded) {
    if (decoded is! Map<String, dynamic>) {
      return null;
    }
    final detail = decoded['detail'];
    if (detail is String) {
      return detail;
    }
    if (detail is List && detail.isNotEmpty) {
      final first = detail.first;
      if (first is Map<String, dynamic> && first['msg'] is String) {
        return first['msg'] as String;
      }
    }
    return detail?.toString();
  }

  static ApiErrorKind _kindFor(int statusCode) {
    if (statusCode == 404) {
      return ApiErrorKind.notFound;
    }
    if (statusCode == 503) {
      return ApiErrorKind.notReady;
    }
    if (statusCode >= 400 && statusCode < 500) {
      return ApiErrorKind.badRequest;
    }
    return ApiErrorKind.server;
  }
}
