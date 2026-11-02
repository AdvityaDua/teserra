import 'dart:async';

import 'package:flutter/material.dart';

import 'runtime/nllb_languages.dart';
import 'runtime/nllb_runtime.dart';
import 'runtime/tiny_whisper_asr_service.dart';

void main() => runApp(const TesseraApp());

class TesseraApp extends StatelessWidget {
  const TesseraApp({super.key});

  @override
  Widget build(BuildContext context) => MaterialApp(
    title: 'Tessera',
    debugShowCheckedModeBanner: false,
    theme: ThemeData(
      useMaterial3: true,
      scaffoldBackgroundColor: const Color(0xfff7f8fa),
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xff155eef)),
      textTheme: const TextTheme(
        headlineLarge: TextStyle(
          fontSize: 38,
          fontWeight: FontWeight.w700,
          letterSpacing: -1.1,
          color: Color(0xff172033),
        ),
      ),
    ),
    home: const TranslationWorkbench(),
  );
}

class TranslationWorkbench extends StatefulWidget {
  const TranslationWorkbench({super.key});

  @override
  State<TranslationWorkbench> createState() => _TranslationWorkbenchState();
}

class _TranslationWorkbenchState extends State<TranslationWorkbench> {
  static final languages = NllbLanguages.all;
  final input = TextEditingController(
    text: 'The fastest path is the one we can measure.',
  );
  NllbLanguage source = languages.firstWhere((l) => l.code == 'eng_Latn');
  NllbLanguage target = languages.firstWhere((l) => l.code == 'spa_Latn');
  String output = 'Run a benchmark to see the translated output.';
  bool running = false;
  String runtimeStatus =
      'Android runtime is ready to download on first benchmark.';
  final runtime = NllbRuntime();
  final asr = TinyWhisperAsrService();
  StreamSubscription<TinyWhisperAsrEvent>? asrSubscription;
  final voiceQueue = <String>[];
  bool listening = false;
  bool translatingVoice = false;
  String liveTranscript = '';
  String committedVoiceText = '';
  String asrStatus = 'Tap the microphone to start local speech input.';
  final voiceTranslations = <String>[];
  double? cpuPercent;
  double? memoryMb;
  DateTime? lastRun;
  List<Stage> stages = const [
    Stage('Input capture', Icons.edit_note_outlined),
    Stage('Tokenization', Icons.data_object_outlined),
    Stage('Encoder', Icons.memory_outlined),
    Stage('Decoder', Icons.auto_awesome_outlined),
    Stage('Detokenization', Icons.text_fields_outlined),
  ];

  @override
  void dispose() {
    input.dispose();
    runtime.dispose();
    asrSubscription?.cancel();
    asr.dispose();
    super.dispose();
  }

  Future<void> toggleListening() async {
    if (listening) {
      await asr.stop();
      if (mounted) setState(() => listening = false);
      return;
    }

    asrSubscription ??= asr.events.listen(_handleAsrEvent);
    try {
      // Preserve anything already in the source box. New confirmed speech is
      // appended after it instead of replacing the previous sentence.
      committedVoiceText = input.text.trim();
      liveTranscript = committedVoiceText;
      await asr.start(nllbLanguageCode: source.code);
      if (mounted) setState(() => listening = true);
    } catch (error) {
      if (mounted) {
        setState(() {
          listening = false;
          asrStatus = 'ASR error: $error';
        });
      }
    }
  }

  void _handleAsrEvent(TinyWhisperAsrEvent event) {
    if (!mounted) return;
    if (event.kind == 'partial') {
      setState(() {
        liveTranscript = _joinVoiceText(committedVoiceText, event.text);
        _setInputText(liveTranscript);
        asrStatus = 'Listening on-device';
      });
    } else if (event.kind == 'sentence') {
      final sentence = event.text.trim();
      if (sentence.isEmpty) return;
      committedVoiceText = _joinVoiceText(committedVoiceText, sentence);
      setState(() {
        liveTranscript = committedVoiceText;
        _setInputText(committedVoiceText);
        asrStatus = 'Translating recognized sentence…';
      });
      voiceQueue.add(sentence);
      _drainVoiceQueue();
    } else if (event.kind == 'status' || event.kind == 'error') {
      setState(() => asrStatus = event.text);
    }
  }

  String _joinVoiceText(String committed, String provisional) {
    final left = committed.trim();
    final right = provisional.trim();
    if (left.isEmpty) return right;
    if (right.isEmpty) return left;
    return '$left $right';
  }

  void _setInputText(String text) {
    input.value = TextEditingValue(
      text: text,
      selection: TextSelection.collapsed(offset: text.length),
    );
  }

  Future<void> _drainVoiceQueue() async {
    if (translatingVoice) return;
    translatingVoice = true;
    while (voiceQueue.isNotEmpty) {
      final sentence = voiceQueue.removeAt(0);
      try {
        final result = await runtime.translate(
          text: sentence,
          sourceLanguage: source.code,
          targetLanguage: target.code,
          onStatus: (status) {
            if (mounted) setState(() => asrStatus = status);
          },
        );
        if (!mounted) return;
        setState(() {
          voiceTranslations.add(result.text);
          output = voiceTranslations.join('\n\n');
          asrStatus = 'Listening for the next sentence';
        });
      } catch (error) {
        if (mounted) setState(() => asrStatus = 'Translation error: $error');
      }
    }
    translatingVoice = false;
  }

  Future<void> runBenchmark() async {
    if (running || input.text.trim().isEmpty) return;
    setState(() {
      running = true;
      cpuPercent = null;
      memoryMb = null;
      output = 'Preparing local Android inference…';
      lastRun = null;
      stages = stages.map((s) => s.copy(status: 'Queued', ms: null)).toList();
    });
    try {
      final result = await runtime.translate(
        text: input.text.trim(),
        sourceLanguage: source.code,
        targetLanguage: target.code,
        onStatus: (status) {
          if (mounted) setState(() => runtimeStatus = status);
        },
      );
      if (!mounted) return;
      final t = result.timings;
      setState(() {
        output = result.text;
        stages = [
          stages[0].copy(status: 'Complete', ms: 0),
          stages[1].copy(status: 'Complete', ms: t['Tokenization']),
          stages[2].copy(status: 'Complete', ms: t['Encoder']),
          stages[3].copy(status: 'Complete', ms: t['Decoder']),
          stages[4].copy(status: 'Complete', ms: t['Detokenization']),
        ];
        running = false;
        lastRun = DateTime.now();
        cpuPercent = result.cpuPercent;
        memoryMb = result.peakMemoryMb;
      });
    } catch (error) {
      if (!mounted) return;
      setState(() {
        output = 'Runtime error: $error';
        runtimeStatus = 'Android runtime error';
        running = false;
      });
    }
  }

  double get totalMs => stages.fold(0, (sum, stage) => sum + (stage.ms ?? 0));

  @override
  Widget build(BuildContext context) => Scaffold(
    body: SafeArea(
      child: LayoutBuilder(
        builder: (context, box) {
          final wide = box.maxWidth >= 960;
          return SingleChildScrollView(
            padding: EdgeInsets.symmetric(
              horizontal: wide ? 56 : 20,
              vertical: wide ? 34 : 20,
            ),
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 1320),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const TopBar(),
                    const SizedBox(height: 54),
                    HeroHeader(lastRun: lastRun),
                    const SizedBox(height: 30),
                    RuntimeNotice(status: runtimeStatus),
                    const SizedBox(height: 22),
                    wide ? wideWorkbench() : stackedWorkbench(),
                    if (liveTranscript.isNotEmpty || listening) ...[
                      const SizedBox(height: 14),
                      VoiceStatus(
                        listening: listening,
                        translating: translatingVoice,
                        transcript: liveTranscript,
                        status: asrStatus,
                      ),
                    ],
                    const SizedBox(height: 24),
                    Telemetry(
                      stages: stages,
                      totalMs: totalMs,
                      running: running,
                      cpuPercent: cpuPercent,
                      memoryMb: memoryMb,
                    ),
                    const SizedBox(height: 22),
                    const FooterNote(),
                  ],
                ),
              ),
            ),
          );
        },
      ),
    ),
  );

  Widget wideWorkbench() => Row(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Expanded(child: inputPanel()),
      const Padding(padding: EdgeInsets.only(top: 166), child: SwapMark()),
      Expanded(child: outputPanel()),
    ],
  );

  Widget stackedWorkbench() => Column(
    children: [
      inputPanel(),
      const SizedBox(height: 14),
      const SwapMark(),
      const SizedBox(height: 14),
      outputPanel(),
    ],
  );

  Widget inputPanel() => TextPanel(
    eyebrow: 'SOURCE / INPUT',
    title: 'Write text to translate',
    language: source,
    onLanguage: (v) => setState(() => source = v),
    footer: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          '${input.text.characters.length} characters',
          style: const TextStyle(color: Color(0xff8791a4), fontSize: 12),
        ),
        OutlinedButton.icon(
          onPressed: running ? null : toggleListening,
          icon: Icon(listening ? Icons.stop : Icons.mic_none, size: 17),
          label: Text(listening ? 'Stop listening' : 'Speak'),
          style: OutlinedButton.styleFrom(
            foregroundColor: listening
                ? const Color(0xffb42318)
                : const Color(0xff155eef),
            side: BorderSide(
              color: listening
                  ? const Color(0xfffda29b)
                  : const Color(0xffb2ccff),
            ),
          ),
        ),
      ],
    ),
    child: TextField(
      controller: input,
      minLines: 7,
      maxLines: 10,
      style: const TextStyle(
        fontSize: 18,
        height: 1.5,
        color: Color(0xff172033),
      ),
      decoration: const InputDecoration(
        hintText: 'Type or paste text here…',
        border: InputBorder.none,
        contentPadding: EdgeInsets.zero,
      ),
    ),
  );

  Widget outputPanel() => TextPanel(
    eyebrow: 'TARGET / OUTPUT',
    title: 'Translated result',
    language: target,
    onLanguage: (v) => setState(() => target = v),
    footer: Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        const Text(
          'Selectable output',
          style: TextStyle(color: Color(0xff8791a4), fontSize: 12),
        ),
        FilledButton.icon(
          onPressed: running ? null : runBenchmark,
          icon: Icon(running ? Icons.sync : Icons.play_arrow, size: 17),
          label: Text(running ? 'Benchmarking…' : 'Run benchmark'),
          style: FilledButton.styleFrom(
            backgroundColor: const Color(0xff155eef),
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(horizontal: 17, vertical: 13),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(9),
            ),
          ),
        ),
      ],
    ),
    child: SelectableText(
      output,
      style: TextStyle(
        fontSize: 18,
        height: 1.5,
        color: output.startsWith('Run') || output.startsWith('Running')
            ? const Color(0xffa1a9b8)
            : const Color(0xff172033),
      ),
    ),
  );
}

class Stage {
  const Stage(this.name, this.icon, {this.status = 'Waiting', this.ms});
  final String name;
  final IconData icon;
  final String status;
  final double? ms;
  Stage copy({String? status, double? ms}) =>
      Stage(name, icon, status: status ?? this.status, ms: ms);
}

class TopBar extends StatelessWidget {
  const TopBar({super.key});
  @override
  Widget build(BuildContext context) => Row(
    children: [
      Container(
        width: 38,
        height: 38,
        decoration: BoxDecoration(
          color: const Color(0xff172033),
          borderRadius: BorderRadius.circular(11),
        ),
        child: const Icon(
          Icons.grid_view_rounded,
          color: Colors.white,
          size: 21,
        ),
      ),
      const SizedBox(width: 12),
      const Text(
        'TESSERA',
        style: TextStyle(
          fontSize: 15,
          fontWeight: FontWeight.w800,
          letterSpacing: 2.4,
          color: Color(0xff172033),
        ),
      ),
      const Spacer(),
      Container(
        padding: const EdgeInsets.symmetric(horizontal: 11, vertical: 7),
        decoration: BoxDecoration(
          color: const Color(0xffeaf8f1),
          borderRadius: BorderRadius.circular(40),
          border: Border.all(color: const Color(0xffc9eed9)),
        ),
        child: const Row(
          children: [
            Icon(Icons.wifi_off_rounded, size: 14, color: Color(0xff15803d)),
            SizedBox(width: 6),
            Text(
              'OFFLINE MODE',
              style: TextStyle(
                color: Color(0xff15803d),
                fontSize: 11,
                fontWeight: FontWeight.w800,
                letterSpacing: .5,
              ),
            ),
          ],
        ),
      ),
    ],
  );
}

class HeroHeader extends StatelessWidget {
  const HeroHeader({super.key, this.lastRun});
  final DateTime? lastRun;
  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.start,
    children: [
      Text(
        'Translation benchmark',
        style: Theme.of(context).textTheme.headlineLarge,
      ),
      const SizedBox(height: 10),
      Row(
        children: [
          const Expanded(
            child: Text(
              'Compare language paths with the model running entirely on-device.',
              style: TextStyle(color: Color(0xff667187), fontSize: 16),
            ),
          ),
          if (lastRun != null)
            Text(
              'Last run ${lastRun!.hour}:${lastRun!.minute.toString().padLeft(2, '0')}',
              style: const TextStyle(color: Color(0xff8791a4), fontSize: 12),
            ),
        ],
      ),
    ],
  );
}

class RuntimeNotice extends StatelessWidget {
  const RuntimeNotice({super.key, required this.status});
  final String status;
  @override
  Widget build(BuildContext context) => Container(
    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
    decoration: BoxDecoration(
      color: const Color(0xffeaf8f1),
      border: Border.all(color: const Color(0xffc9eed9)),
      borderRadius: BorderRadius.circular(10),
    ),
    child: Row(
      children: [
        Icon(Icons.memory_outlined, size: 18, color: Color(0xff15803d)),
        SizedBox(width: 10),
        Expanded(
          child: Text(
            status,
            style: TextStyle(color: Color(0xff166534), fontSize: 13),
          ),
        ),
      ],
    ),
  );
}

class TextPanel extends StatelessWidget {
  const TextPanel({
    super.key,
    required this.eyebrow,
    required this.title,
    required this.language,
    required this.onLanguage,
    required this.child,
    required this.footer,
  });
  final String eyebrow;
  final String title;
  final NllbLanguage language;
  final ValueChanged<NllbLanguage> onLanguage;
  final Widget child;
  final Widget footer;
  @override
  Widget build(BuildContext context) => Container(
    height: 330,
    padding: const EdgeInsets.fromLTRB(22, 20, 22, 16),
    decoration: BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(14),
      border: Border.all(color: const Color(0xffe1e5ec)),
    ),
    child: Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    eyebrow,
                    style: const TextStyle(
                      color: Color(0xff8791a4),
                      fontSize: 11,
                      fontWeight: FontWeight.w800,
                      letterSpacing: 1.1,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    title,
                    style: const TextStyle(
                      color: Color(0xff172033),
                      fontSize: 17,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                ],
              ),
            ),
            DropdownButtonHideUnderline(
              child: DropdownButton<NllbLanguage>(
                value: language,
                isExpanded: false,
                borderRadius: BorderRadius.circular(10),
                items: _TranslationWorkbenchState.languages
                    .map(
                      (v) => DropdownMenuItem(
                        value: v,
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 220),
                          child: Text(v.name, overflow: TextOverflow.ellipsis),
                        ),
                      ),
                    )
                    .toList(),
                onChanged: (v) {
                  if (v != null) onLanguage(v);
                },
              ),
            ),
          ],
        ),
        const SizedBox(height: 18),
        const Divider(height: 1, color: Color(0xffe8ebf0)),
        const SizedBox(height: 18),
        Expanded(child: child),
        const SizedBox(height: 14),
        const Divider(height: 1, color: Color(0xffe8ebf0)),
        const SizedBox(height: 12),
        footer,
      ],
    ),
  );
}

class VoiceStatus extends StatelessWidget {
  const VoiceStatus({
    super.key,
    required this.listening,
    required this.translating,
    required this.transcript,
    required this.status,
  });

  final bool listening;
  final bool translating;
  final String transcript;
  final String status;

  @override
  Widget build(BuildContext context) => Container(
    width: double.infinity,
    padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 14),
    decoration: BoxDecoration(
      color: const Color(0xffeef4ff),
      borderRadius: BorderRadius.circular(12),
      border: Border.all(color: const Color(0xffc7d7fe)),
    ),
    child: Row(
      children: [
        Icon(
          listening ? Icons.mic : Icons.mic_off,
          color: const Color(0xff155eef),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                translating ? 'Translating locally' : status,
                style: const TextStyle(
                  color: Color(0xff173b82),
                  fontWeight: FontWeight.w700,
                ),
              ),
              if (transcript.isNotEmpty) ...[
                const SizedBox(height: 3),
                Text(
                  transcript,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(color: Color(0xff40557d)),
                ),
              ],
            ],
          ),
        ),
        if (translating)
          const SizedBox(
            width: 18,
            height: 18,
            child: CircularProgressIndicator(strokeWidth: 2),
          ),
      ],
    ),
  );
}

class SwapMark extends StatelessWidget {
  const SwapMark({super.key});
  @override
  Widget build(BuildContext context) => Container(
    margin: const EdgeInsets.symmetric(horizontal: 14),
    width: 38,
    height: 38,
    decoration: BoxDecoration(
      color: const Color(0xffeff4ff),
      borderRadius: BorderRadius.circular(10),
      border: Border.all(color: const Color(0xffd5e2ff)),
    ),
    child: const Icon(
      Icons.compare_arrows_rounded,
      color: Color(0xff155eef),
      size: 21,
    ),
  );
}

class Telemetry extends StatelessWidget {
  const Telemetry({
    super.key,
    required this.stages,
    required this.totalMs,
    required this.running,
    required this.cpuPercent,
    required this.memoryMb,
  });
  final List<Stage> stages;
  final double totalMs;
  final bool running;
  final double? cpuPercent;
  final double? memoryMb;
  @override
  Widget build(BuildContext context) {
    final complete = stages.where((s) => s.ms != null).length;
    return Container(
      padding: const EdgeInsets.all(22),
      decoration: BoxDecoration(
        color: const Color(0xff172033),
        borderRadius: BorderRadius.circular(14),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'RUNTIME TELEMETRY',
                      style: TextStyle(
                        color: Color(0xff98a5bd),
                        fontSize: 11,
                        fontWeight: FontWeight.w800,
                        letterSpacing: 1.2,
                      ),
                    ),
                    SizedBox(height: 6),
                    Text(
                      'Inference path',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
              ),
              Metric(
                label: 'END-TO-END',
                value: complete == stages.length
                    ? '${totalMs.toStringAsFixed(1)} ms'
                    : '—',
                color: const Color(0xff6ea2ff),
              ),
              const SizedBox(width: 32),
              Metric(
                label: 'STAGES',
                value: '$complete / ${stages.length}',
                color: const Color(0xff7ee2a8),
              ),
            ],
          ),
          const SizedBox(height: 20),
          const Divider(color: Color(0xff303b50), height: 1),
          Padding(
            padding: const EdgeInsets.only(top: 16, bottom: 2),
            child: Row(
              children: [
                Metric(
                  label: 'PEAK RSS',
                  value: memoryMb == null
                      ? '—'
                      : '${memoryMb!.toStringAsFixed(0)} MB',
                  color: const Color(0xffffc46b),
                ),
                const SizedBox(width: 32),
                Metric(
                  label: 'CPU CORES',
                  value: cpuPercent == null
                      ? '—'
                      : (cpuPercent! / 100).toStringAsFixed(1),
                  color: const Color(0xffc4a7ff),
                ),
              ],
            ),
          ),
          ...stages.map((stage) => StageRow(stage: stage, totalMs: totalMs)),
          if (running)
            const Padding(
              padding: EdgeInsets.only(top: 10),
              child: LinearProgressIndicator(
                minHeight: 3,
                backgroundColor: Color(0xff303b50),
                color: Color(0xff6ea2ff),
              ),
            ),
        ],
      ),
    );
  }
}

class Metric extends StatelessWidget {
  const Metric({
    super.key,
    required this.label,
    required this.value,
    required this.color,
  });
  final String label;
  final String value;
  final Color color;
  @override
  Widget build(BuildContext context) => Column(
    crossAxisAlignment: CrossAxisAlignment.end,
    children: [
      Text(
        label,
        style: const TextStyle(
          color: Color(0xff98a5bd),
          fontSize: 10,
          letterSpacing: .8,
          fontWeight: FontWeight.w700,
        ),
      ),
      const SizedBox(height: 5),
      Text(
        value,
        style: TextStyle(
          color: color,
          fontSize: 18,
          fontWeight: FontWeight.w700,
        ),
      ),
    ],
  );
}

class StageRow extends StatelessWidget {
  const StageRow({super.key, required this.stage, required this.totalMs});
  final Stage stage;
  final double totalMs;
  @override
  Widget build(BuildContext context) {
    final complete = stage.ms != null;
    final width = complete && totalMs > 0
        ? (stage.ms! / totalMs).clamp(.04, 1.0)
        : 0.0;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 13),
      child: Row(
        children: [
          Icon(
            stage.icon,
            size: 18,
            color: stage.status == 'Running'
                ? const Color(0xff6ea2ff)
                : const Color(0xff98a5bd),
          ),
          const SizedBox(width: 12),
          SizedBox(
            width: 145,
            child: Text(
              stage.name,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 13,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          Expanded(
            child: ClipRRect(
              borderRadius: BorderRadius.circular(10),
              child: LinearProgressIndicator(
                value: stage.status == 'Running' ? null : width,
                minHeight: 6,
                backgroundColor: const Color(0xff303b50),
                color: complete
                    ? const Color(0xff6ea2ff)
                    : const Color(0xff475873),
              ),
            ),
          ),
          const SizedBox(width: 16),
          SizedBox(
            width: 76,
            child: Text(
              complete ? '${stage.ms!.toStringAsFixed(1)} ms' : stage.status,
              textAlign: TextAlign.right,
              style: const TextStyle(color: Color(0xffb9c8e5), fontSize: 12),
            ),
          ),
        ],
      ),
    );
  }
}

class FooterNote extends StatelessWidget {
  const FooterNote({super.key});
  @override
  Widget build(BuildContext context) => const Row(
    children: [
      Icon(Icons.info_outline, size: 16, color: Color(0xff8791a4)),
      SizedBox(width: 8),
      Expanded(
        child: Text(
          'Timings are measured on the device. Peak RSS includes native model memory; CPU cores is process usage across cores.',
          style: TextStyle(color: Color(0xff8791a4), fontSize: 12),
        ),
      ),
    ],
  );
}
