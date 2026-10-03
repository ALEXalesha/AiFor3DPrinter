# Data and model sources / Источники данных и модели

Every source used for training is listed here with its licence. Anything without a licence,
with an unclear licence or "non-commercial only" was skipped (list at the end).
Raw frames are never committed: they stay in `%LOCALAPPDATA%\AiForPrinter\dataset`.

Здесь записан каждый источник, на котором учится сеть, и его лицензия. Без лицензии, с
неясной лицензией или «только некоммерческое» - не берётся (список в конце). Сами кадры в
репозиторий не кладутся, они лежат в `%LOCALAPPDATA%\AiForPrinter\dataset`.

## Used / Используется

| Source | What | Licence | How it is used |
|---|---|---|---|
| Own timelapses of the author's Elegoo Centauri Carbon (115 prints, 79 models, 12 871 frames) | normal printing | own data, not published | `normal` class; split by print model into train / val / test |
| [DasKunststoffZentrumSKZ / Errors_Additive_Manufacturing_Plattform_Cam](https://huggingface.co/datasets/DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Plattform_Cam) (Hugging Face), project 16IS23050B | top-down camera, YOLO boxes incl. *Spaghetti*; 980 of 3 145 frames taken (700 train / 120 val / 160 test, authors' split), downscaled to 960x540 | **CC BY-SA 4.0** | real spaghetti: whole frames as `spaghetti` and cut-out spaghetti boxes recoloured and pasted onto Centauri frames |
| Synthetic failures drawn by `dataset/synth.py` over own frames | spaghetti, blob, detached, no extrusion | MIT (this repository) | failure classes |
| torchvision `mobilenet_v3_large` ImageNet weights (`IMAGENET1K_V2`) | pretrained backbone | BSD-3-Clause (torchvision) | fine-tuned |
| Frames marked "✅ Всё нормально" in Telegram (`feedback/normal`) | the author's own camera | own data, not published | `normal` class, train only |

Attribution (CC BY-SA 4.0): "Errors_Additive_Manufacturing_Plattform_Cam" by Das Kunststoff-Zentrum SKZ,
part of the project "Künstliche Intelligenz zur automatisierten Fehlerkorrektur in der additiven
Fertigung" (16IS23050B), https://huggingface.co/datasets/DasKunststoffZentrumSKZ/Errors_Additive_Manufacturing_Plattform_Cam.
The images were downscaled; spaghetti boxes were cut, recoloured and pasted onto other frames.

**ShareAlike and the model weights.** Whether trained weights are an "adapted work" of the training
images is legally unsettled. To stay on the safe side, the trained weights `model/aifp.onnx` are
released under **CC BY-SA 4.0** with the attribution above. The code of this repository is MIT.

**ShareAlike и веса.** Являются ли веса сети переработкой обучающих картинок - вопрос не решённый.
Чтобы не гадать, веса `model/aifp.onnx` выкладываются под **CC BY-SA 4.0** с указанием источника
выше. Код репозитория - MIT.

## Skipped / Не взято

| Source | Why |
|---|---|
| Obico / The Spaghetti Detective models and code | AGPL-3.0: not copied (ideas only) |
| Roboflow Universe "Spaghetti 3D", "3D printing failure", "3D printing flaws" (CC BY 4.0) | download needs an account and API key |
| Hugging Face `Javiai/failures-3D-print` | licence "unknown" |
| Hugging Face `Masamsa/3d-print-failure-detection` | no licence |
| Zenodo 19200472 (PLA/ABS layerwise defects) | access restricted |
| Cambridge CAXTON dataset | nozzle close-up camera, ~1 TB; viewpoint does not match |
| `joanna-/3D-Printing-Data` (CC BY 4.0) | sensor data, no images |
