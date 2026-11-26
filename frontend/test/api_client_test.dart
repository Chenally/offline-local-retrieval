import 'dart:convert';

import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/src/api/models.dart';
import 'package:frontend/src/api/retrieval_api.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';

HttpRetrievalApi apiWith(MockClientHandler handler) {
  return HttpRetrievalApi(
    baseUri: Uri.parse('http://127.0.0.1:8765'),
    client: MockClient(handler),
  );
}

http.Response jsonResponse(Object body, [int status = 200]) => http.Response(
      jsonEncode(body),
      status,
      headers: <String, String>{'content-type': 'application/json'},
    );

void main() {
  test('search sends filters and parses explained results', () async {
    late Map<String, dynamic> sent;
    final api = apiWith((request) async {
      expect(request.method, 'POST');
      expect(request.url.toString(), 'http://127.0.0.1:8765/search');
      sent = jsonDecode(request.body) as Map<String, dynamic>;
      return jsonResponse(<String, dynamic>{
        'query': 'budget',
        'took_ms': 12.5,
        'warnings': <String>[],
        'results': <Map<String, dynamic>>[
          <String, dynamic>{
            'document_id': 'abc',
            'kind': 'text',
            'file_name': 'budget.pdf',
            'extension': '.pdf',
            'paths': <String>['/docs/budget.pdf'],
            'score': 0.9,
            'channel_scores': <String, dynamic>{
              'keyword': <String, dynamic>{
                'raw': 3.2,
                'normalized': 1.0,
                'weight': 0.3,
                'contribution': 0.3,
                'rank': 1,
              },
            },
            'snippet': 'Budget overview',
            'page_number': 2,
            'chunk_id': 'abc:0',
          },
        ],
      });
    });

    final response = await api.search(
      const SearchRequest(
        query: 'budget',
        topK: 20,
        modes: <SearchMode>{SearchMode.keyword},
        extensions: <String>['pdf'],
      ),
    );

    expect(sent['top_k'], 20);
    expect(sent['modes'], <String>['keyword']);
    expect(sent['extensions'], <String>['pdf']);
    expect(response.results.single.fileName, 'budget.pdf');
    expect(response.results.single.pageNumber, 2);
    expect(response.results.single.channelScores[SearchMode.keyword]!.weight, 0.3);
  });

  test('index status and jobs are parsed', () async {
    final api = apiWith((request) async {
      if (request.url.path == '/index/status') {
        return jsonResponse(<String, dynamic>{
          'roots': <String>['/docs'],
          'file_count': 4,
          'document_count': 4,
          'text_file_count': 3,
          'image_file_count': 1,
          'text_vector_count': 9,
          'image_vector_count': 1,
          'active_job': null,
        });
      }
      return jsonResponse(<String, dynamic>{
        'job_id': 'j1',
        'operation': 'index',
        'directory': '/docs',
        'state': 'running',
        'processed': 2,
        'total': 4,
        'current_path': '/docs/a.txt',
        'report': null,
        'removed_count': 0,
        'error': null,
      }, 202);
    });

    final status = await api.indexStatus();
    final job = await api.startIndexing('/docs');

    expect(status.roots, <String>['/docs']);
    expect(status.imageFileCount, 1);
    expect(job.state, JobState.running);
    expect(job.progress, 0.5);
    expect(job.isFinished, isFalse);
  });

  test('HTTP errors become typed ApiExceptions with readable messages', () async {
    final api = apiWith((request) async {
      if (request.url.path == '/search') {
        return jsonResponse(<String, dynamic>{'detail': 'Search query cannot be empty'}, 400);
      }
      return jsonResponse(<String, dynamic>{'detail': 'models missing'}, 503);
    });

    await expectLater(
      api.search(const SearchRequest(query: ' ')),
      throwsA(
        isA<ApiException>()
            .having((e) => e.kind, 'kind', ApiErrorKind.badRequest)
            .having((e) => e.userMessage, 'message', 'Search query cannot be empty'),
      ),
    );
    await expectLater(
      api.readiness(),
      throwsA(isA<ApiException>().having((e) => e.kind, 'kind', ApiErrorKind.notReady)),
    );
  });

  test('validation errors use the first message from FastAPI', () async {
    final api = apiWith((request) async => jsonResponse(<String, dynamic>{
          'detail': <Map<String, dynamic>>[
            <String, dynamic>{'msg': 'top_k must be at most 100'},
          ],
        }, 422));

    await expectLater(
      api.search(const SearchRequest(query: 'x', topK: 500)),
      throwsA(isA<ApiException>().having((e) => e.message, 'message', 'top_k must be at most 100')),
    );
  });

  test('connection failures are reported as unreachable', () async {
    final api = apiWith((request) async => throw http.ClientException('refused'));

    expect(await api.isAlive(), isFalse);
    await expectLater(
      api.indexStatus(),
      throwsA(isA<ApiException>().having((e) => e.kind, 'kind', ApiErrorKind.unreachable)),
    );
  });

  test('base URI can change at runtime', () async {
    final urls = <String>[];
    final api = apiWith((request) async {
      urls.add(request.url.toString());
      return jsonResponse(<String, dynamic>{'status': 'ok'});
    });

    await api.isAlive();
    api.baseUri = Uri.parse('http://localhost:9000/');
    await api.isAlive();

    expect(urls, <String>[
      'http://127.0.0.1:8765/health/live',
      'http://localhost:9000/health/live',
    ]);
  });
}
