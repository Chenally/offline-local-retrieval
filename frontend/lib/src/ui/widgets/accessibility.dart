import 'package:material_ui/material_ui.dart';
import 'package:flutter/semantics.dart';

/// Asks the platform screen reader (NVDA, Narrator, VoiceOver) to read
/// [message] without moving keyboard focus.
void announce(BuildContext context, String message) {
  if (message.isEmpty) {
    return;
  }
  SemanticsService.sendAnnouncement(
    View.of(context),
    message,
    Directionality.of(context),
  );
}

/// Shows a status line and announces it whenever its text changes, so screen
/// reader users hear the outcome of searches and indexing jobs (WCAG 4.1.3).
class AnnouncedStatus extends StatefulWidget {
  const AnnouncedStatus(
    this.message, {
    super.key,
    this.style,
    this.visible = true,
  });

  final String message;
  final TextStyle? style;

  /// When false the message is still announced but not drawn, for cases
  /// where another widget (such as an [ErrorPanel]) already shows it.
  final bool visible;

  @override
  State<AnnouncedStatus> createState() => _AnnouncedStatusState();
}

class _AnnouncedStatusState extends State<AnnouncedStatus> {
  @override
  void didUpdateWidget(AnnouncedStatus oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.message != oldWidget.message) {
      announce(context, widget.message);
    }
  }

  @override
  Widget build(BuildContext context) {
    if (widget.message.isEmpty || !widget.visible) {
      return const SizedBox.shrink();
    }
    return Text(widget.message, style: widget.style);
  }
}

/// A section heading exposed to assistive technology as a header, so screen
/// reader users can jump between sections (WCAG 1.3.1, 2.4.6).
class SectionHeading extends StatelessWidget {
  const SectionHeading(this.text, {super.key, this.large = false});

  final String text;
  final bool large;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context).textTheme;
    return Semantics(
      header: true,
      child: Text(text, style: large ? theme.headlineSmall : theme.titleMedium),
    );
  }
}

/// Draws a visible outline while [child] or one of its descendants has
/// keyboard focus (WCAG 2.4.7).
class FocusOutline extends StatefulWidget {
  const FocusOutline({
    super.key,
    required this.child,
    this.borderRadius = const BorderRadius.all(Radius.circular(12)),
  });

  final Widget child;
  final BorderRadius borderRadius;

  @override
  State<FocusOutline> createState() => _FocusOutlineState();
}

class _FocusOutlineState extends State<FocusOutline> {
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final color = Theme.of(context).colorScheme.primary;
    return Focus(
      canRequestFocus: false,
      skipTraversal: true,
      // Only tracks focus for the outline; adds nothing for screen readers.
      includeSemantics: false,
      onFocusChange: (focused) => setState(() => _focused = focused),
      child: DecoratedBox(
        position: DecorationPosition.foreground,
        decoration: BoxDecoration(
          borderRadius: widget.borderRadius,
          border: Border.all(
            color: _focused ? color : Colors.transparent,
            width: 3,
          ),
        ),
        child: widget.child,
      ),
    );
  }
}

/// A recoverable error with an explanation and a retry action (WCAG 3.3.1).
class ErrorPanel extends StatelessWidget {
  const ErrorPanel({
    super.key,
    required this.message,
    this.onRetry,
    this.retryLabel = 'Try again',
  });

  final String message;
  final VoidCallback? onRetry;
  final String retryLabel;

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Card(
      color: scheme.errorContainer,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Wrap(
          spacing: 16,
          runSpacing: 12,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: <Widget>[
            Row(
              mainAxisSize: MainAxisSize.min,
              children: <Widget>[
                Icon(Icons.error_outline, color: scheme.onErrorContainer),
                const SizedBox(width: 12),
                Flexible(
                  child: Text(
                    message,
                    style: TextStyle(color: scheme.onErrorContainer),
                  ),
                ),
              ],
            ),
            if (onRetry != null)
              FilledButton.icon(
                onPressed: onRetry,
                icon: const Icon(Icons.refresh),
                label: Text(retryLabel),
              ),
          ],
        ),
      ),
    );
  }
}
