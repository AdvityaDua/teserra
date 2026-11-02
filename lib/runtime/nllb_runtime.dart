import 'dart:typed_data';

import 'package:dart_sentencepiece_tokenizer/dart_sentencepiece_tokenizer.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

import 'model_bundle_service.dart';
import 'nllb_languages.dart';
import 'process_metrics.dart';

class NllbTimings {
  const NllbTimings({required this.values});
  final Map<String, double> values;
  double operator [](String key) => values[key] ?? 0;
  double get total => values.values.fold(0, (sum, value) => sum + value);
}

class NllbTranslationResult {
  const NllbTranslationResult({
    required this.text,
    required this.timings,
    required this.inputTokens,
    required this.outputTokens,
    required this.peakMemoryMb,
    required this.cpuPercent,
  });
  final String text;
  final NllbTimings timings;
  final int inputTokens;
  final int outputTokens;
  final double peakMemoryMb;
  final double cpuPercent;
}

class NllbRuntime {
  NllbRuntime({ModelBundleService? bundles})
    : _bundles = bundles ?? ModelBundleService();

  static const eos = 2;
  static final languages = {
    for (final language in NllbLanguages.all) language.code: language.code,
  };

  final ModelBundleService _bundles;
  final OnnxRuntime _ort = OnnxRuntime();
  OrtSession? _encoder;
  OrtSession? _decoder;
  SentencePieceTokenizer? _tokenizer;

  bool get isReady => _encoder != null && _decoder != null;

  Future<NllbTranslationResult> translate({
    required String text,
    required String sourceLanguage,
    required String targetLanguage,
    void Function(String status)? onStatus,
    void Function(double progress)? onDownloadProgress,
  }) async {
    final timings = <String, double>{};
    final overall = Stopwatch()..start();
    final resourceStart = ProcessSnapshot.capture();
    var peakMemoryMb = resourceStart.rssMb;
    await _ensureReady(onStatus, onDownloadProgress);
    peakMemoryMb = _peakMemory(peakMemoryMb);

    final tokenTimer = Stopwatch()..start();
    final tokenizer = _tokenizer!;
    final sourceCode = languages[sourceLanguage] ?? sourceLanguage;
    final targetCode = languages[targetLanguage] ?? targetLanguage;
    final pieces = tokenizer.encode(text, addSpecialTokens: false).ids;
    final sourceId = tokenizer.convertTokensToIds([sourceCode]).single;
    final targetId = tokenizer.convertTokensToIds([targetCode]).single;
    final inputIds = <int>[sourceId, ...pieces, eos];
    tokenTimer.stop();
    timings['Tokenization'] = tokenTimer.elapsedMicroseconds / 1000;
    onStatus?.call('Running encoder');

    final encoderTimer = Stopwatch()..start();
    final encoderIds = await OrtValue.fromList(Int64List.fromList(inputIds), [
      1,
      inputIds.length,
    ]);
    final encoderMask = await OrtValue.fromList(
      Int64List.fromList(List.filled(inputIds.length, 1)),
      [1, inputIds.length],
    );
    final encoderOutputs = await _encoder!.run({
      'input_ids': encoderIds,
      'attention_mask': encoderMask,
    });
    await encoderIds.dispose();
    final hidden = encoderOutputs['last_hidden_state']!;
    encoderTimer.stop();
    timings['Encoder'] = encoderTimer.elapsedMicroseconds / 1000;
    peakMemoryMb = _peakMemory(peakMemoryMb);

    onStatus?.call('Running decoder');
    final decoderTimer = Stopwatch()..start();
    final firstInput = await OrtValue.fromList(Int64List.fromList([eos]), [
      1,
      1,
    ]);
    final firstOutputs = await _decoder!.run({
      'input_ids': firstInput,
      'encoder_attention_mask': encoderMask,
      'encoder_hidden_states': hidden,
    });
    await firstInput.dispose();
    final generated = <int>[targetId];
    var next = targetId;

    for (final value in firstOutputs.values) {
      await value.dispose();
    }
    while (generated.length < 96 && next != eos) {
      final input = await OrtValue.fromList(
        Int64List.fromList([eos, ...generated]),
        [1, generated.length + 1],
      );
      final outputs = await _decoder!.run({
        'input_ids': input,
        'encoder_attention_mask': encoderMask,
        'encoder_hidden_states': hidden,
      });
      await input.dispose();
      final logits = outputs['logits']!;
      next = await logits.argMax();
      generated.add(next);
      await logits.dispose();
      for (final value in outputs.values) {
        if (value != logits) await value.dispose();
      }
      peakMemoryMb = _peakMemory(peakMemoryMb);
    }
    peakMemoryMb = _peakMemory(peakMemoryMb);
    decoderTimer.stop();
    timings['Decoder'] = decoderTimer.elapsedMicroseconds / 1000;
    await encoderMask.dispose();
    await hidden.dispose();
    for (final value in encoderOutputs.values) {
      if (value != hidden) await value.dispose();
    }

    final detokenTimer = Stopwatch()..start();
    final translated = tokenizer
        .decode(generated, skipSpecialTokens: true)
        .trim();
    detokenTimer.stop();
    timings['Detokenization'] = detokenTimer.elapsedMicroseconds / 1000;
    overall.stop();
    timings['End-to-end'] = overall.elapsedMicroseconds / 1000;
    final resourceEnd = ProcessSnapshot.capture();
    onStatus?.call('Translation complete');
    return NllbTranslationResult(
      text: translated,
      timings: NllbTimings(values: timings),
      inputTokens: inputIds.length,
      outputTokens: generated.length,
      peakMemoryMb: peakMemoryMb > resourceEnd.rssMb
          ? peakMemoryMb
          : resourceEnd.rssMb,
      cpuPercent: resourceEnd.cpuPercentSince(resourceStart),
    );
  }

  double _peakMemory(double current) {
    final rss = ProcessSnapshot.capture().rssMb;
    return rss > current ? rss : current;
  }

  Future<void> _ensureReady(
    void Function(String status)? onStatus,
    void Function(double progress)? onDownloadProgress,
  ) async {
    if (isReady) return;
    final paths = await _bundles.ensureAndroidBundle(
      onStatus: onStatus,
      onProgress: onDownloadProgress,
    );
    _tokenizer = await TokenizerJsonLoader.fromJsonFile(paths.tokenizer);
    final options = OrtSessionOptions(
      intraOpNumThreads: 4,
      interOpNumThreads: 1,
      useArena: false,
      providers: [OrtProvider.XNNPACK, OrtProvider.CPU],
    );
    _encoder = await _ort.createSession(paths.encoder, options: options);
    _decoder = await _ort.createSession(paths.decoder, options: options);
    onStatus?.call('Android ONNX Runtime ready');
  }

  Future<void> dispose() async {
    await _encoder?.close();
    await _decoder?.close();
    _bundles.dispose();
  }
}
