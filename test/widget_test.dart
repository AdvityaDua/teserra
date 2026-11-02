import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:tessera/main.dart';

void main() {
  testWidgets('translation workbench renders the Android benchmark shell', (
    tester,
  ) async {
    await tester.pumpWidget(const TesseraApp());

    expect(find.text('Translation benchmark'), findsOneWidget);
    expect(find.text('Write text to translate'), findsOneWidget);
    expect(find.text('Translated result'), findsOneWidget);
    expect(find.text('Run benchmark'), findsOneWidget);

    await tester.scrollUntilVisible(
      find.text('Run benchmark'),
      400,
      scrollable: find.byType(Scrollable).first,
    );
    expect(find.text('RUNTIME TELEMETRY'), findsOneWidget);
    expect(
      find.text('Android runtime is ready to download on first benchmark.'),
      findsOneWidget,
    );
  });
}
