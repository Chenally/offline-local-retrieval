import 'package:material_ui/material_ui.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:frontend/src/ui/theme.dart';

/// Text/background pairs used by the interface. WCAG 1.4.3 requires 4.5:1
/// for normal text; high-contrast themes target 7:1 (WCAG 1.4.6).
Map<String, List<Color>> textPairs(ColorScheme s) => <String, List<Color>>{
      'onSurface/surface': <Color>[s.onSurface, s.surface],
      'onSurfaceVariant/surface': <Color>[s.onSurfaceVariant, s.surface],
      'onPrimary/primary': <Color>[s.onPrimary, s.primary],
      'onSecondaryContainer/secondaryContainer': <Color>[
        s.onSecondaryContainer,
        s.secondaryContainer,
      ],
      'onErrorContainer/errorContainer': <Color>[s.onErrorContainer, s.errorContainer],
      'error/surface': <Color>[s.error, s.surface],
    };

void main() {
  final themes = <String, ThemeData>{
    'light': buildTheme(brightness: Brightness.light, highContrast: false),
    'dark': buildTheme(brightness: Brightness.dark, highContrast: false),
    'high-contrast light': buildTheme(brightness: Brightness.light, highContrast: true),
    'high-contrast dark': buildTheme(brightness: Brightness.dark, highContrast: true),
  };

  themes.forEach((name, theme) {
    final scheme = theme.colorScheme;
    final minimum = name.startsWith('high-contrast') ? 7.0 : 4.5;

    test('$name theme text contrast is at least $minimum:1', () {
      textPairs(scheme).forEach((pair, colors) {
        expect(
          contrastRatio(colors[0], colors[1]),
          greaterThanOrEqualTo(minimum),
          reason: '$pair in $name',
        );
      });
    });

    test('$name theme focus outline contrast is at least 3:1 (WCAG 1.4.11)', () {
      expect(contrastRatio(focusColorFor(scheme), scheme.surface), greaterThanOrEqualTo(3.0));
    });

    test('$name theme uses 48 px minimum targets', () {
      expect(theme.materialTapTargetSize, MaterialTapTargetSize.padded);
      final size = theme.filledButtonTheme.style!.minimumSize!.resolve(<WidgetState>{});
      expect(size, const Size(minTargetSize, minTargetSize));
    });
  });

  test('contrast ratio matches WCAG reference values', () {
    expect(contrastRatio(Colors.black, Colors.white), closeTo(21, 0.01));
    expect(contrastRatio(Colors.white, Colors.white), closeTo(1, 0.01));
  });
}
