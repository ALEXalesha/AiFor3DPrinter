# AiForPrinter

**A neural network watches the camera of an Elegoo Centauri Carbon while it prints and notices failures: spaghetti, the part detached from the bed, a blob of plastic on the nozzle, printing "into the air" with no extrusion. The alert arrives in Telegram as a photo with a heat map of where the failure is and two buttons, "⏸ Pause" and "✅ All good". The program never stops a print, and it pauses without a human only if that is explicitly turned on.**

[Русская версия](README.ru.md) - the full manual; this file summarises it. The window and the Telegram texts are in Russian and English.

> Status on 2026-10-03: plan steps 1-7 are done (service, network, dataset, window, bot buttons). A week of observation on real prints (step 8) and the release (step 9) are ahead. The n8n bot with the alert buttons was deployed on 2026-10-03 and the service runs hidden (`tools\watch-hidden.vbs`, stop with `tools\stop-watch.ps1`).

## What it is

The printer camera and the n8n Telegram bot already existed (`camera_server.py` on port 8099 gives a snapshot, the status and pause). What was missing were eyes: a 9-hour print turns into spaghetti in the first hour and nobody sees it until morning. This program is eyes, not hands.

| | |
| --- | --- |
| When it looks | only while the status is "printing" (Centauri code 13). Heating, calibration, pause - the camera is not touched |
| How often | one shot every 15 s (5-300, configurable) |
| When it alerts | 3 of the last 4 shots show a failure with confidence 0.7 or more; then 10 minutes of quiet. After "All good" that failure type stays quiet until the print ends |
| What it sends | a photo with the heat map and a box around the hottest spot, the failure type, confidence, file and layer |
| What it does by itself | nothing. Pause is a human's button. Auto-pause exists but is off until the false-alarm numbers are good |
| Stopping a print | never. The camera_server client has no such request; laws and mutation tests check it |
| Where the network runs | onnxruntime on the CPU, 2 threads, ~5 ms per frame; PyTorch is only needed for training |

## Printer safety: "never stopprint"

The client (`watcher/camera_client.py`) knows exactly three paths - `/snap`, `/status`, `/pause` - in a frozen set. A less obvious hole was closed as well: camera_server routes on `self.path.split("?")[0]`, so a configured address like `http://127.0.0.1:8099/stopprint?x=` would have turned a snapshot request into `/stopprint?x=/snap` and stopped the print. The camera_server address is therefore accepted only as `http://host:port`, with no path, query, fragment or credentials. Property tests run the service against a fake camera_server over random sequences of statuses, failures and network outputs; there is never a `/stopprint`, and without auto-pause never a `/pause`. `tools/mutate.py` breaks 23 rules in a copy of the code and checks that the right test fails: 23 of 23 are caught.

## Model

MobileNetV3-Large (torchvision, BSD-3, ImageNet-pretrained), fine-tuned on an RTX 5060 Ti. The head is a 1x1 convolution plus averaging, so the per-class map is exactly Grad-CAM for that head and is exported from ONNX as a second output.

Full passport: [model/PASSPORT.md](model/PASSPORT.md) (in Russian). The headline number is **false alarms per hour of normal printing**, measured on prints whose models the network never saw, in real print time, with every timelapse frame held for its whole layer (the worst case for the 3-of-4 rule): **0.41 per hour** (18 alarms in 43.8 hours, 23 test prints, all 18 in three unusual prints). If every false alarm is answered with "✅ All good" - the service then mutes that failure type until the end of the print - it is **0.11 per hour**.

The first trained network gave 0.87 false alarms per hour; 34 of 38 came from three prints with several red parts on the bed. Synthetic spaghetti is "coloured stuff scattered on the bed", and normal training frames almost always showed one part. The fix (`dataset/synth.py::extra_parts`) adds upright copies of the part to normal frames, and normal frames now make up 55% of an epoch; that halved the alarms. The rest needs data from the camera itself, which the "All good" button collects.

The failures in the evaluation are synthetic, drawn over real frames of the test prints, so the per-type accuracy is optimistic. Normal frames are real, so the false-alarm rate is honest - but measured on timelapses, not on live shots. Hence the observation week, with auto-pause off.

## The missed detachment of 2026-10-03

A thin early-layer part came off the bed, the nozzle dragged it around and finally hit the toolhead with it; the service sent nothing. Its journal shows it was watching every attempt and never raised an alarm, but it kept neither per-shot scores nor frames, so what the network saw cannot be reconstructed (the cancelled print has no timelapse). Now every shot is logged with its scores, frames are kept per print in a size-capped local store (`main.py review` shows the timeline, `--rescore` re-runs a newer network), the first 10 layers are shot every 5 s with a more sensitive rule (0.5, 2 of 3), and a background-motion metric is logged for calibration on live shots. On synthetic "part dragged across the bed" sequences from 5% of the print the alarm comes in a median of 8 s on 23 of 23 test prints - but the old network already did that, so the synthetic drag does not look like the real event; the next real case will be reviewable.

## Data

Own Centauri timelapses (115 prints, 79 models, 12 871 frames, never committed) for normal printing; 980 frames of the SKZ dataset "Errors_Additive_Manufacturing_Plattform_Cam" (Hugging Face, **CC BY-SA 4.0**) for real spaghetti; synthetic failures drawn by `dataset/synth.py`. Train/val/test are split by print model, not by frame. Sources, licences and what was skipped (Obico - AGPL, Roboflow - needs an account, unlicensed sets): [dataset/LICENSES.md](dataset/LICENSES.md).

The trained weights `model/aifp.onnx` are released under **CC BY-SA 4.0** (they were trained partly on CC BY-SA images); the code is MIT.

## Running

```powershell
uv venv -p 3.13 .venv
uv pip install -r requirements-dev.txt
.venv\Scripts\python main.py ui        # window
.venv\Scripts\python main.py watch     # service (or tools\watch.cmd)
.venv\Scripts\python main.py check     # one shot now, read-only
.venv\Scripts\python main.py review    # what the service saw: prints and per-shot scores
.venv\Scripts\python -m pytest -q      # tests (no printer, no camera_server, no bot)
.venv\Scripts\python tools\mutate.py   # mutation check
```

Training commands are in the [Russian README](README.ru.md#обучение).
