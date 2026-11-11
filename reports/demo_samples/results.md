# Demo samples: results

- Face model: DFDC-winning EfficientNet-B7 (Seferbekov, 2020)
- Voice model: project voice model (audio_asvspoof_3a53cf51)
- Threshold: 0.50; video verdict = fuse_video (face OR voice; a borderline score cannot overrule a confident one) unless a fusion model is loaded; face scores average each face with its mirror image
- Correct: **10 / 11**

| sample                            | kind   | truth   | category              | face P(fake)   | voice P(fake)   | P(fake)   | verdict   | correct   |   seconds |
|:----------------------------------|:-------|:--------|:----------------------|:---------------|:----------------|:----------|:----------|:----------|----------:|
| video/real_actor.mp4              | video  | REAL    | RealVideo-RealAudio   | 1.3%           | 51.5%           | 1.3%      | REAL      | ✓         |       9.4 |
| video/fake_faceswap.mp4           | video  | FAKE    | FakeVideo-RealAudio   | 99.0%          | 0.0%            | 99.0%     | FAKE      | ✓         |       9.1 |
| video/fake_cloned_voice.mp4       | video  | FAKE    | RealVideo-FakeAudio   | 1.3%           | 95.8%           | 95.8%     | FAKE      | ✓         |       8.8 |
| video/fake_faceswap_and_voice.mp4 | video  | FAKE    | FakeVideo-FakeAudio   | 99.0%          | 99.6%           | 99.6%     | FAKE      | ✓         |       8.8 |
| audio/real_ljspeech.wav           | audio  | REAL    | bona fide             | –              | 92.2%           | 92.2%     | FAKE      | ✗         |       0.1 |
| audio/fake_vits.wav               | audio  | FAKE    | VITS                  | –              | 100.0%          | 100.0%    | FAKE      | ✓         |       0.2 |
| audio/fake_tacotron2.wav          | audio  | FAKE    | Tacotron 2 + HiFi-GAN | –              | 100.0%          | 100.0%    | FAKE      | ✓         |       0.2 |
| audio/fake_glowtts.wav            | audio  | FAKE    | Glow-TTS + HiFi-GAN   | –              | 100.0%          | 100.0%    | FAKE      | ✓         |       0.1 |
| image/real_face.jpg               | image  | REAL    | real photo            | 0.7%           | –               | 0.7%      | REAL      | ✓         |       0.4 |
| image/fake_faceswap_front.jpg     | image  | FAKE    | face swap             | 99.0%          | –               | 99.0%     | FAKE      | ✓         |       0.4 |
| image/fake_faceswap_gesture.jpg   | image  | FAKE    | face swap             | 98.9%          | –               | 98.9%     | FAKE      | ✓         |       0.4 |
