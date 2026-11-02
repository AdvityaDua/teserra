import 'dart:async';
import 'dart:io';

import 'package:whisper_cpp_flutter_plus/whisper_cpp_flutter_plus.dart';

class TinyWhisperAsrService {
  static const modelEndpoint =
      'http://127.0.0.1:8080/v1/models/whisper-tiny/download';

  final _events = StreamController<TinyWhisperAsrEvent>.broadcast();
  final _models = WhisperModelManager();
  WhisperEngine? _engine;
  WhisperStreamTask? _task;
  StreamSubscription<WhisperStreamUpdate>? _updates;
  int _confirmedSegments = 0;
  String _pendingSentence = '';

  Stream<TinyWhisperAsrEvent> get events => _events.stream;
  bool get isRunning => _task != null;

  Future<void> start({required String nllbLanguageCode}) async {
    if (_task != null) return;
    _events.add(
      const TinyWhisperAsrEvent.status('Preparing Tiny Whisper locally…'),
    );
    final model = await _ensureModel();
    _engine ??= await WhisperEngine.load(
      model.path,
      config: const WhisperConfig(backend: WhisperBackend.automatic),
    );
    _confirmedSegments = 0;
    _pendingSentence = '';
    final task = await _engine!.transcribeMicrophone(
      options: TranscribeOptions(
        language: _whisperLanguage(nllbLanguageCode),
        tokenTimestamps: false,
        noTimestamps: true,
        noContext: true,
      ).withPerformanceMode(WhisperPerformanceMode.responsive),
      config: const WhisperStreamConfig(
        updateInterval: Duration(seconds: 2),
        windowDuration: Duration(seconds: 20),
        confirmationLag: Duration(seconds: 2),
      ),
    );
    _task = task;
    _updates = task.updates.listen(_handleUpdate);
    _events.add(const TinyWhisperAsrEvent.status('Listening on-device'));
  }

  Future<File> _ensureModel() async {
    final descriptor = WhisperModelCatalog.tiny;
    try {
      final cached = await _models.findCatalogModel(descriptor);
      if (cached != null) return cached;
    } on FormatException {
      final invalid = await _models.find(descriptor.fileName);
      if (invalid != null) await invalid.delete();
    }
    _events.add(
      const TinyWhisperAsrEvent.status(
        'Downloading Tiny Whisper from Tessera…',
      ),
    );
    await for (final progress in _models.download(
      Uri.parse(modelEndpoint),
      descriptor.fileName,
      sha256Hex: descriptor.sha256,
      allowInsecureHttp: true,
    )) {
      final fraction = progress.fraction;
      if (fraction != null) {
        _events.add(
          TinyWhisperAsrEvent.status(
            'Downloading Tiny Whisper ${(fraction * 100).toStringAsFixed(0)}%',
          ),
        );
      }
    }
    final downloaded = await _models.findCatalogModel(descriptor);
    if (downloaded == null) {
      throw const FileSystemException('Tiny Whisper model was not installed');
    }
    return downloaded;
  }

  void _handleUpdate(WhisperStreamUpdate update) {
    if (update.partialText.trim().isNotEmpty) {
      _events.add(TinyWhisperAsrEvent.partial(update.partialText.trim()));
    }
    _consumeConfirmedSegments(update.confirmedSegments);
  }

  void _consumeConfirmedSegments(List<WhisperSegment> segments) {
    while (_confirmedSegments < segments.length) {
      final segment = segments[_confirmedSegments++].text.trim();
      if (segment.isNotEmpty) _consumeConfirmed(segment);
    }
  }

  void _consumeConfirmed(String text) {
    _pendingSentence = '$_pendingSentence $text'.trim();
    final matches = RegExp(r'(.+?[.!?])(?:\s+|$)').allMatches(_pendingSentence);
    var consumed = 0;
    for (final match in matches) {
      final sentence = match.group(1)?.trim();
      if (sentence != null && sentence.isNotEmpty) {
        _events.add(TinyWhisperAsrEvent.sentence(sentence));
      }
      consumed = match.end;
    }
    if (consumed > 0) {
      _pendingSentence = _pendingSentence.substring(consumed).trim();
    }
  }

  String _whisperLanguage(String nllbCode) {
    const codes = <String, String>{
      'eng': 'en',
      'hin': 'hi',
      'spa': 'es',
      'fra': 'fr',
      'deu': 'de',
      'ita': 'it',
      'por': 'pt',
      'rus': 'ru',
      'zho': 'zh',
      'jpn': 'ja',
      'kor': 'ko',
      'arb': 'ar',
      'pes': 'fa',
      'ben': 'bn',
      'guj': 'gu',
      'kan': 'kn',
      'mal': 'ml',
      'mar': 'mr',
      'pan': 'pa',
      'tam': 'ta',
      'tel': 'te',
      'urd': 'ur',
      'tur': 'tr',
      'nld': 'nl',
      'pol': 'pl',
      'ukr': 'uk',
      'vie': 'vi',
      'tha': 'th',
      'heb': 'he',
      'ind': 'id',
      'swe': 'sv',
      'dan': 'da',
      'fin': 'fi',
      'nor': 'no',
      'cat': 'ca',
      'ces': 'cs',
      'hun': 'hu',
      'ron': 'ro',
      'bul': 'bg',
      'ell': 'el',
      'srp': 'sr',
      'hrv': 'hr',
      'afr': 'af',
      'amh': 'am',
      'yor': 'yo',
      'ibo': 'ig',
      'hau': 'ha',
      'som': 'so',
      'zul': 'zu',
      'mya': 'my',
      'khm': 'km',
      'lao': 'lo',
      'sin': 'si',
    };
    return codes[nllbCode.split('_').first] ?? 'auto';
  }

  Future<void> stop() async {
    final task = _task;
    if (task == null) return;
    final finalUpdate = await task.stop();
    await _updates?.cancel();
    _task = null;
    // The final update can contain the complete confirmed transcript, including
    // segments already delivered through the stream. Consume only new segments
    // so stopping the microphone cannot enqueue duplicate sentences.
    _consumeConfirmedSegments(finalUpdate.confirmedSegments);
    if (_pendingSentence.isNotEmpty) {
      _events.add(TinyWhisperAsrEvent.sentence(_pendingSentence));
      _pendingSentence = '';
    }
    _events.add(const TinyWhisperAsrEvent.status('Microphone paused'));
  }

  Future<void> dispose() async {
    await stop();
    await _updates?.cancel();
    _engine?.dispose();
    _models.close();
    await _events.close();
  }
}

class TinyWhisperAsrEvent {
  const TinyWhisperAsrEvent._(this.kind, this.text);

  const TinyWhisperAsrEvent.partial(String text) : this._('partial', text);
  const TinyWhisperAsrEvent.sentence(String text) : this._('sentence', text);
  const TinyWhisperAsrEvent.status(String text) : this._('status', text);
  const TinyWhisperAsrEvent.error(String text) : this._('error', text);

  final String kind;
  final String text;
}
