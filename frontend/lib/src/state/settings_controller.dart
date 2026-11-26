import 'dart:convert';

import 'package:material_ui/material_ui.dart';
import 'package:shared_preferences/shared_preferences.dart';

const String defaultServiceUrl = 'http://127.0.0.1:8765';
const double minTextScale = 1.0;
const double maxTextScale = 2.0;
const List<int> resultCountOptions = <int>[10, 20, 50];

/// User-adjustable preferences. Immutable; changes go through
/// [SettingsController.update].
@immutable
class AppSettings {
  const AppSettings({
    this.themeMode = ThemeMode.system,
    this.highContrast = false,
    this.textScale = 1.0,
    this.serviceUrl = defaultServiceUrl,
    this.resultsPerSearch = 10,
  });

  factory AppSettings.fromMap(Map<String, Object?> map) {
    const defaults = AppSettings();
    final themeName = map['themeMode'];
    final scale = map['textScale'];
    final results = map['resultsPerSearch'];
    final url = map['serviceUrl'];

    return AppSettings(
      themeMode: ThemeMode.values.firstWhere(
        (mode) => mode.name == themeName,
        orElse: () => defaults.themeMode,
      ),
      highContrast: map['highContrast'] == true,
      textScale: scale is num
          ? scale.toDouble().clamp(minTextScale, maxTextScale).toDouble()
          : defaults.textScale,
      serviceUrl: url is String && url.isNotEmpty ? url : defaults.serviceUrl,
      resultsPerSearch: results is int && resultCountOptions.contains(results)
          ? results
          : defaults.resultsPerSearch,
    );
  }

  final ThemeMode themeMode;
  final bool highContrast;
  final double textScale;
  final String serviceUrl;
  final int resultsPerSearch;

  AppSettings copyWith({
    ThemeMode? themeMode,
    bool? highContrast,
    double? textScale,
    String? serviceUrl,
    int? resultsPerSearch,
  }) {
    return AppSettings(
      themeMode: themeMode ?? this.themeMode,
      highContrast: highContrast ?? this.highContrast,
      textScale: textScale ?? this.textScale,
      serviceUrl: serviceUrl ?? this.serviceUrl,
      resultsPerSearch: resultsPerSearch ?? this.resultsPerSearch,
    );
  }

  Map<String, Object?> toMap() {
    return <String, Object?>{
      'themeMode': themeMode.name,
      'highContrast': highContrast,
      'textScale': textScale,
      'serviceUrl': serviceUrl,
      'resultsPerSearch': resultsPerSearch,
    };
  }
}

abstract class SettingsStore {
  Future<Map<String, Object?>> load();
  Future<void> save(Map<String, Object?> values);
}

class SharedPreferencesSettingsStore implements SettingsStore {
  static const String _key = 'app_settings_v1';

  @override
  Future<Map<String, Object?>> load() async {
    final preferences = await SharedPreferences.getInstance();
    final raw = preferences.getString(_key);

    if (raw == null) {
      return <String, Object?>{};
    }

    try {
      final decoded = jsonDecode(raw);
      return decoded is Map<String, dynamic> ? decoded : <String, Object?>{};
    } on FormatException {
      return <String, Object?>{};
    }
  }

  @override
  Future<void> save(Map<String, Object?> values) async {
    final preferences = await SharedPreferences.getInstance();
    await preferences.setString(_key, jsonEncode(values));
  }
}

class MemorySettingsStore implements SettingsStore {
  MemorySettingsStore([Map<String, Object?>? initial])
      : values = <String, Object?>{...?initial};

  Map<String, Object?> values;

  @override
  Future<Map<String, Object?>> load() async => <String, Object?>{...values};

  @override
  Future<void> save(Map<String, Object?> values) async {
    this.values = <String, Object?>{...values};
  }
}

class SettingsController extends ChangeNotifier {
  SettingsController(this._store);

  final SettingsStore _store;
  AppSettings _value = const AppSettings();

  AppSettings get value => _value;

  Future<void> load() async {
    _value = AppSettings.fromMap(await _store.load());
    notifyListeners();
  }

  Future<void> update(AppSettings next) async {
    _value = next;
    notifyListeners();
    await _store.save(next.toMap());
  }

  Future<void> setThemeMode(ThemeMode mode) =>
      update(_value.copyWith(themeMode: mode));

  Future<void> setHighContrast(bool enabled) =>
      update(_value.copyWith(highContrast: enabled));

  Future<void> setTextScale(double scale) => update(
        _value.copyWith(
          textScale: scale.clamp(minTextScale, maxTextScale).toDouble(),
        ),
      );

  Future<void> setResultsPerSearch(int count) =>
      update(_value.copyWith(resultsPerSearch: count));

  Future<void> setServiceUrl(String url) =>
      update(_value.copyWith(serviceUrl: url.trim()));
}
