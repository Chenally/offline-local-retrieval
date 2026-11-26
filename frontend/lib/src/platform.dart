import 'dart:io';

import 'package:file_selector/file_selector.dart';
import 'package:url_launcher/url_launcher.dart';

/// Desktop integrations, injected so widget tests can replace them.
class PlatformActions {
  const PlatformActions({
    required this.pickDirectory,
    required this.openPath,
    required this.revealPath,
  });

  /// Shows the system folder picker; returns null if the user cancels.
  final Future<String?> Function() pickDirectory;

  /// Opens a file with its default application.
  final Future<bool> Function(String path) openPath;

  /// Opens the folder that contains a file.
  final Future<bool> Function(String path) revealPath;

  static PlatformActions desktop() {
    return PlatformActions(
      pickDirectory: () => getDirectoryPath(confirmButtonText: 'Index this folder'),
      openPath: (path) => launchUrl(Uri.file(path)),
      revealPath: (path) => launchUrl(Uri.directory(File(path).parent.path)),
    );
  }
}
