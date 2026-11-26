import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/src/api/models.dart';
import 'package:frontend/src/api/retrieval_api.dart';
import 'package:frontend/src/state/library_view_model.dart';
import 'package:frontend/src/state/search_view_model.dart';
import 'package:frontend/src/state/settings_controller.dart';

import 'support/fakes.dart';

void main() {
  group('SearchViewModel', () {
    test('sends modes, file types and result count', () async {
      final api = FakeRetrievalApi();
      final vm = SearchViewModel(api)
        ..toggleMode(SearchMode.imageSemantic)
        ..toggleFileType(FileTypeFilter.pdf);

      await vm.search('  budget  ', topK: 20);

      final request = api.searchRequests.single;
      expect(request.query, 'budget');
      expect(request.topK, 20);
      expect(request.modes, <SearchMode>{SearchMode.keyword, SearchMode.textSemantic});
      expect(request.extensions, <String>['pdf']);
      expect(vm.results, hasLength(1));
      expect(vm.statusMessage, '1 result for "budget".');
      expect(vm.loading, isFalse);
    });

    test('keeps at least one search method enabled', () {
      final vm = SearchViewModel(FakeRetrievalApi())
        ..toggleMode(SearchMode.keyword)
        ..toggleMode(SearchMode.textSemantic)
        ..toggleMode(SearchMode.imageSemantic);

      expect(vm.modes, <SearchMode>{SearchMode.imageSemantic});
      expect(vm.statusMessage, contains('At least one'));
    });

    test('empty query does not call the service', () async {
      final api = FakeRetrievalApi();
      final vm = SearchViewModel(api);

      await vm.search('   ', topK: 10);

      expect(api.searchRequests, isEmpty);
      expect(vm.statusMessage, 'Type something to search for.');
    });

    test('service errors are exposed with a readable message', () async {
      final api = FakeRetrievalApi()
        ..searchError = const ApiException(ApiErrorKind.unreachable, 'refused');
      final vm = SearchViewModel(api);

      await vm.search('budget', topK: 10);

      expect(vm.error, isNotNull);
      expect(vm.statusMessage, contains('not responding'));
      expect(vm.results, isEmpty);
    });

    test('warnings and empty results are described', () async {
      final api = FakeRetrievalApi(results: <SearchResult>[])
        ..warnings = <String>['image_semantic search unavailable'];
      final vm = SearchViewModel(api);

      await vm.search('cat', topK: 10);

      expect(vm.statusMessage, 'No results for "cat". Some search methods are unavailable.');
    });
  });

  group('LibraryViewModel', () {
    test('adding a folder follows the job and refreshes the index', () async {
      final api = FakeRetrievalApi()
        ..jobSequence = <IndexJob>[
          sampleJob(state: JobState.running, processed: 2, total: 3),
          sampleJob(),
        ];
      final vm = LibraryViewModel(api, pollInterval: Duration.zero);

      await vm.addFolder('/library');

      expect(api.indexedDirectories, <String>['/library']);
      expect(vm.isBusy, isFalse);
      expect(vm.lastFinishedJob!.state, JobState.succeeded);
      expect(vm.statusMessage, startsWith('Finished indexing library: 2 new or changed'));
      expect(vm.folders, <String>['/library']);
    });

    test('removing a folder reports the number of files', () async {
      final api = FakeRetrievalApi();
      final vm = LibraryViewModel(api, pollInterval: Duration.zero);

      await vm.removeFolder('/library');

      expect(api.removedDirectories, <String>['/library']);
      expect(vm.statusMessage, 'Removed library from the index (3 files).');
    });

    test('failed jobs are described', () {
      final job = IndexJob(
        jobId: 'x',
        operation: 'index',
        directory: r'C:\Users\me\Docs',
        state: JobState.failed,
        processed: 0,
        total: 0,
        error: 'disk full',
      );

      expect(
        LibraryViewModel.describeFinishedJob(job),
        'Indexing Docs failed: disk full.',
      );
    });

    test('refresh errors are kept for display', () async {
      final api = FakeRetrievalApi()
        ..statusError = const ApiException(ApiErrorKind.unreachable, 'refused');
      final vm = LibraryViewModel(api, pollInterval: Duration.zero);

      await vm.refresh();

      expect(vm.error, isNotNull);
      expect(vm.loading, isFalse);
    });
  });

  group('SettingsController', () {
    test('persists and restores settings', () async {
      final store = MemorySettingsStore();
      final controller = SettingsController(store);

      await controller.setThemeMode(ThemeMode.dark);
      await controller.setHighContrast(true);
      await controller.setTextScale(5);
      await controller.setResultsPerSearch(50);

      final restored = SettingsController(store);
      await restored.load();

      expect(restored.value.themeMode, ThemeMode.dark);
      expect(restored.value.highContrast, isTrue);
      expect(restored.value.textScale, maxTextScale);
      expect(restored.value.resultsPerSearch, 50);
    });

    test('ignores invalid stored values', () async {
      final controller = SettingsController(
        MemorySettingsStore(<String, Object?>{
          'themeMode': 'neon',
          'textScale': 'big',
          'resultsPerSearch': 7,
          'serviceUrl': '',
        }),
      );

      await controller.load();

      expect(controller.value.themeMode, ThemeMode.system);
      expect(controller.value.textScale, 1.0);
      expect(controller.value.resultsPerSearch, 10);
      expect(controller.value.serviceUrl, defaultServiceUrl);
    });
  });
}
