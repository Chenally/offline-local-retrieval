import 'package:material_ui/material_ui.dart';

const Color _seed = Color(0xFF1B5E9E);

/// Minimum interactive size (WCAG 2.5.5 / platform guidelines: 44–48 px).
const double minTargetSize = 48;

/// Width of the focus outline drawn around focused controls (WCAG 2.4.7).
const double focusOutlineWidth = 3;

ThemeData buildTheme({
  required Brightness brightness,
  required bool highContrast,
}) {
  final scheme = highContrast
      ? highContrastScheme(brightness)
      : ColorScheme.fromSeed(seedColor: _seed, brightness: brightness);

  final focusSide = WidgetStateProperty.resolveWith<BorderSide?>((states) {
    if (states.contains(WidgetState.focused)) {
      return BorderSide(color: focusColorFor(scheme), width: focusOutlineWidth);
    }
    return null;
  });

  final buttonStyle = ButtonStyle(
    minimumSize: const WidgetStatePropertyAll<Size>(
      Size(minTargetSize, minTargetSize),
    ),
    side: focusSide,
  );

  return ThemeData(
    colorScheme: scheme,
    useMaterial3: true,
    materialTapTargetSize: MaterialTapTargetSize.padded,
    visualDensity: VisualDensity.standard,
    filledButtonTheme: FilledButtonThemeData(style: buttonStyle),
    outlinedButtonTheme: OutlinedButtonThemeData(style: buttonStyle),
    textButtonTheme: TextButtonThemeData(style: buttonStyle),
    elevatedButtonTheme: ElevatedButtonThemeData(style: buttonStyle),
    iconButtonTheme: IconButtonThemeData(style: buttonStyle),
    segmentedButtonTheme: SegmentedButtonThemeData(style: buttonStyle),
    inputDecorationTheme: InputDecorationTheme(
      border: const OutlineInputBorder(),
      focusedBorder: OutlineInputBorder(
        borderSide: BorderSide(
          color: focusColorFor(scheme),
          width: focusOutlineWidth,
        ),
      ),
    ),
    cardTheme: CardThemeData(
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: highContrast
            ? BorderSide(color: scheme.onSurface)
            : BorderSide.none,
      ),
    ),
    dividerTheme: DividerThemeData(
      color: highContrast ? scheme.onSurface : null,
    ),
  );
}

/// Color used for focus outlines: high contrast against the surface.
Color focusColorFor(ColorScheme scheme) => scheme.primary;

/// Black-on-white or white-on-black palettes with saturated accents. All
/// text pairs exceed 7:1 contrast (WCAG 1.4.6, enhanced).
ColorScheme highContrastScheme(Brightness brightness) {
  if (brightness == Brightness.dark) {
    return const ColorScheme(
      brightness: Brightness.dark,
      primary: Color(0xFFFFEB3B),
      onPrimary: Color(0xFF000000),
      primaryContainer: Color(0xFFFFEB3B),
      onPrimaryContainer: Color(0xFF000000),
      secondary: Color(0xFF80F0FF),
      onSecondary: Color(0xFF000000),
      secondaryContainer: Color(0xFF00363D),
      onSecondaryContainer: Color(0xFFFFFFFF),
      tertiary: Color(0xFFFFEB3B),
      onTertiary: Color(0xFF000000),
      error: Color(0xFFFF9E9E),
      onError: Color(0xFF000000),
      errorContainer: Color(0xFF5C0000),
      onErrorContainer: Color(0xFFFFFFFF),
      surface: Color(0xFF000000),
      onSurface: Color(0xFFFFFFFF),
      surfaceContainerLowest: Color(0xFF000000),
      surfaceContainerLow: Color(0xFF000000),
      surfaceContainer: Color(0xFF0A0A0A),
      surfaceContainerHigh: Color(0xFF141414),
      surfaceContainerHighest: Color(0xFF1E1E1E),
      onSurfaceVariant: Color(0xFFFFFFFF),
      outline: Color(0xFFFFFFFF),
      outlineVariant: Color(0xFFBDBDBD),
    );
  }

  return const ColorScheme(
    brightness: Brightness.light,
    primary: Color(0xFF00007A),
    onPrimary: Color(0xFFFFFFFF),
    primaryContainer: Color(0xFF00007A),
    onPrimaryContainer: Color(0xFFFFFFFF),
    secondary: Color(0xFF004D40),
    onSecondary: Color(0xFFFFFFFF),
    secondaryContainer: Color(0xFFD6F5EF),
    onSecondaryContainer: Color(0xFF000000),
    tertiary: Color(0xFF00007A),
    onTertiary: Color(0xFFFFFFFF),
    error: Color(0xFF9B0000),
    onError: Color(0xFFFFFFFF),
    errorContainer: Color(0xFFFFE0E0),
    onErrorContainer: Color(0xFF000000),
    surface: Color(0xFFFFFFFF),
    onSurface: Color(0xFF000000),
    surfaceContainerLowest: Color(0xFFFFFFFF),
    surfaceContainerLow: Color(0xFFFFFFFF),
    surfaceContainer: Color(0xFFF5F5F5),
    surfaceContainerHigh: Color(0xFFEEEEEE),
    surfaceContainerHighest: Color(0xFFE6E6E6),
    onSurfaceVariant: Color(0xFF000000),
    outline: Color(0xFF000000),
    outlineVariant: Color(0xFF424242),
  );
}

/// WCAG relative-luminance contrast ratio between two opaque colors.
double contrastRatio(Color a, Color b) {
  final la = a.computeLuminance();
  final lb = b.computeLuminance();
  final lighter = la > lb ? la : lb;
  final darker = la > lb ? lb : la;
  return (lighter + 0.05) / (darker + 0.05);
}
