# Demo samples

Small labelled clips for trying the app and the CLI. Each type has one real sample and two or
three fakes. Ground truth is in [`samples.json`](samples.json); the app shows it next to the verdict
when a sample is analysed. Rebuild everything from the original sources with
`python scripts/make_demo_samples.py`.

These clips are a sanity check, not a benchmark: eleven files say little about accuracy. The audio is
outside the voice model's training data (ASVspoof 2019). The two DFDC clips come from the DFDC preview
set; the face model was trained on the later full DFDC set, and an overlap of actors or clips cannot be
ruled out. Current results, including the two the models get wrong:
[`reports/demo_samples/results.md`](../reports/demo_samples/results.md).

## Video

| File | Truth | Category | What it is |
|---|---|---|---|
| `video/real_actor.mp4` | real | RealVideo-RealAudio | Unedited DFDC recording (`lynaeydofd`) |
| `video/fake_faceswap.mp4` | fake | FakeVideo-RealAudio | DFDC face swap (`mqzvfufzoq`): another actor's face, original voice |
| `video/fake_cloned_voice.mp4` | fake | RealVideo-FakeAudio | `real_actor.mp4` with its soundtrack replaced by VITS speech |
| `video/fake_faceswap_and_voice.mp4` | fake | FakeVideo-FakeAudio | `fake_faceswap.mp4` with its soundtrack replaced by Glow-TTS + HiFi-GAN speech |

The last two are built by `scripts/make_demo_samples.py` to cover the two fake-voice categories of
FakeAVCeleb. The lips and the voice do not match in them; this project does not check lip sync, so
only the voice model can catch them.

All videos are re-encoded to 720p H.264 (10 s).

## Audio

Same sentence in every file: *"that not more than one bottle of wine or one quart of beer could be
issued at one time. No account was taken of the amount of liquors admitted in one day,"*

| File | Truth | System |
|---|---|---|
| `audio/real_ljspeech.wav` | real | LJ Speech recording (LJ Speech is a public-domain audiobook corpus) |
| `audio/fake_vits.wav` | fake | VITS, end-to-end TTS trained on the LJ Speech voice |
| `audio/fake_tacotron2.wav` | fake | Tacotron 2 + HiFi-GAN vocoder |
| `audio/fake_glowtts.wav` | fake | Glow-TTS + HiFi-GAN vocoder |

The fakes imitate the same speaker as the real recording, which is what a voice clone does.

## Image

| File | Truth | What it is |
|---|---|---|
| `image/real_face.jpg` | real | Frame of the unedited DFDC recording |
| `image/fake_faceswap_front.jpg` | fake | Frame of the DFDC face swap, facing the camera |
| `image/fake_faceswap_gesture.jpg` | fake | Another frame of the same face swap, mid-gesture |

The two fake images come from one clip; no second openly redistributable face swap was available.

## Sources and licences

- **DFDC preview set**: B. Dolhansky et al., *The Deepfake Detection Challenge (DFDC) Preview
  Dataset*, arXiv:1910.08854, 2019. Recorded with paid actors who agreed to the use of their likeness
  for deepfake research. The two clips are the ones shipped with
  [polimi-ispl/icpr2020dfdc](https://github.com/polimi-ispl/icpr2020dfdc) (`notebook/samples/`).
  Use them for research and demonstration only, under the DFDC terms of use.
- **VITS samples**: J. Kim, J. Kong, J. Son, *Conditional Variational Autoencoder with Adversarial
  Learning for End-to-End Text-to-Speech*, ICML 2021. Files from
  [jaywalnut310/vits-demo](https://github.com/jaywalnut310/vits-demo) (`wavs/ss_01_*.wav`,
  `ss_03_gh.wav`). The ground-truth recording is from LJ Speech (K. Ito and L. Johnson, 2017),
  public domain.
