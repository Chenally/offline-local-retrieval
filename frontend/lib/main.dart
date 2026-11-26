import 'package:material_ui/material_ui.dart';

import 'src/api/retrieval_api.dart';
import 'src/app.dart';
import 'src/debug/semantics_logger.dart';
import 'src/platform.dart';
import 'src/state/settings_controller.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  if (kA11yDebug) {
    SemanticsLogger.install();
  }

  final settings = SettingsController(SharedPreferencesSettingsStore());
  await settings.load();

  final api = HttpRetrievalApi(baseUri: Uri.parse(settings.value.serviceUrl));

  runApp(
    OfflineRetrievalApp(
      settings: settings,
      api: api,
      platform: PlatformActions.desktop(),
    ),
  );
}
