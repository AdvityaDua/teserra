import 'dart:io';

import 'package:archive/archive_io.dart';
import 'package:http/http.dart' as http;
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

class ModelBundlePaths {
  const ModelBundlePaths({required this.root});

  final String root;

  String get encoder => p.join(root, 'onnx', 'encoder_model.onnx');
  String get decoder => p.join(root, 'onnx', 'decoder_model.onnx');
  String get decoderWithPast =>
      p.join(root, 'onnx', 'decoder_with_past_model.onnx');
  String get tokenizer => p.join(root, 'source', 'tokenizer.json');
}

class ModelBundleService {
  ModelBundleService({http.Client? client}) : _client = client ?? http.Client();

  static const bundleUrl =
      'http://127.0.0.1:8080/v1/models/android-onnx-int8/download';
  final http.Client _client;

  Future<ModelBundlePaths> ensureAndroidBundle({
    void Function(String status)? onStatus,
    void Function(double progress)? onProgress,
  }) async {
    final support = await getApplicationSupportDirectory();
    // v4 is the first bundle produced from a true FP32 export.
    final root = p.join(
      support.path,
      'tessera',
      'models',
      'android-onnx-int8-v4',
    );
    final paths = ModelBundlePaths(root: root);
    if (await _isComplete(paths)) {
      onStatus?.call('Android model already downloaded');
      return paths;
    }

    final staging = Directory('$root.staging');
    if (await staging.exists()) await staging.delete(recursive: true);
    await staging.create(recursive: true);
    final zipPath = p.join(staging.path, 'bundle.zip');

    onStatus?.call('Downloading Android model');
    final response = await _client.send(
      http.Request('GET', Uri.parse(bundleUrl)),
    );
    if (response.statusCode != 200) {
      throw HttpException('Model service returned HTTP ${response.statusCode}');
    }
    final total = response.contentLength ?? 0;
    var received = 0;
    final sink = File(zipPath).openWrite();
    try {
      await for (final chunk in response.stream) {
        received += chunk.length;
        sink.add(chunk);
        if (total > 0) onProgress?.call(received / total);
      }
    } finally {
      await sink.close();
    }

    onStatus?.call('Extracting Android model');
    final extractedRoot = Directory(p.join(staging.path, 'extracted'));
    await extractedRoot.create(recursive: true);
    await extractFileToDisk(zipPath, extractedRoot.path);
    if (!await File(p.join(extractedRoot.path, 'manifest.json')).exists()) {
      throw const FormatException('Downloaded model bundle has no manifest');
    }
    await File(zipPath).delete();
    if (await Directory(root).exists()) {
      await Directory(root).delete(recursive: true);
    }
    await extractedRoot.rename(root);
    await staging.delete(recursive: true);
    onProgress?.call(1);
    onStatus?.call('Android model ready');
    return paths;
  }

  Future<bool> _isComplete(ModelBundlePaths paths) async => await Future.wait([
    File(paths.encoder).exists(),
    File(paths.decoder).exists(),
    File(paths.decoderWithPast).exists(),
    File(paths.tokenizer).exists(),
  ]).then((values) => values.every((value) => value));

  void dispose() => _client.close();
}
