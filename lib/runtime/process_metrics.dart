import 'dart:io';

class ProcessSnapshot {
  const ProcessSnapshot({required this.rssMb, required this.processTicks, required this.capturedAtMicros});

  final double rssMb;
  final int processTicks;
  final int capturedAtMicros;

  static ProcessSnapshot capture() {
    final status = File('/proc/self/status').readAsStringSync();
    final rssLine = status.split('\n').firstWhere(
      (line) => line.startsWith('VmRSS:'),
      orElse: () => 'VmRSS: 0 kB',
    );
    final rssKb = int.tryParse(RegExp(r'\d+').firstMatch(rssLine)?.group(0) ?? '0') ?? 0;

    final stat = File('/proc/self/stat').readAsStringSync();
    final closingParen = stat.lastIndexOf(')');
    final statFields = closingParen == -1 ? <String>[] : stat.substring(closingParen + 2).trim().split(RegExp(r'\s+'));
    final processTicks = statFields.length > 12
        ? (int.tryParse(statFields[11]) ?? 0) + (int.tryParse(statFields[12]) ?? 0)
        : 0;
    return ProcessSnapshot(
      rssMb: rssKb / 1024,
      processTicks: processTicks,
      capturedAtMicros: DateTime.now().microsecondsSinceEpoch,
    );
  }

  double cpuPercentSince(ProcessSnapshot start) {
    final processDelta = processTicks - start.processTicks;
    final elapsedMicros = capturedAtMicros - start.capturedAtMicros;
    if (processDelta <= 0 || elapsedMicros <= 0) return 0;
    // Android's Linux kernels expose process times in 1/100-second ticks.
    // This reports CPU use as a percentage of one core and does not require
    // the app to read the protected system-wide /proc/stat file.
    return processDelta / 100 * 1000000 / elapsedMicros * 100;
  }
}
