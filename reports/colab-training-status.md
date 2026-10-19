# Colab training status

Active notebook (Mr. Homes, `tks2005homes@gmail.com`):
https://colab.research.google.com/drive/1d_BLDQL7bWBZNg8OPbX817iiSy6KlxE2

The earlier Tanishq-account notebook is preserved:
https://colab.research.google.com/drive/1MZX5OxU35eN7qGQoUJlXN1hJER_p2t1b

The reviewed source was uploaded and installed successfully in Colab on 2026-10-11.
The actual setup output reports PyTorch `2.11.0+cu130` and `Tesla T4`.
Source bundle SHA-256: `3a53cf5154dc675eb36409708d1ce66a7654590c982ea90b5b6fa013c7648125`.

The user requested switching to the Homes account. The reviewed notebook is now
uploaded there, and setup again verified PyTorch and Tesla T4 successfully.
Drive is now successfully mounted under the Homes account. The Edinburgh endpoint
returned HTTP 403 from Colab. The downloader now falls back to the official
Zenodo mirror and verifies the published MD5 before extraction. The mirror is
actively downloading the 7,640,952,520-byte archive. The audio training cell is
visibly queued behind preparation; training has not started and no real-data
accuracy or trained checkpoint is verified yet.

The active notebook embeds the downloader patch and records its SHA-256 alongside
the original source bundle hash. The local notebook preserves that exact cell.
Evidence: `reports/colab-download-training-queued.jpg`.

The notebook downloads the public ASVspoof 2019 LA archive, prepares up to 2,500
examples per class per official split, trains the audio branch for up to 15 epochs,
and saves checkpoints, loss curves, predictions, and held-out metrics in
`MyDrive/deepfake_project`. These will be subset results, not full-benchmark scores.

FakeAVCeleb multimodal training remains dependent on approved raw data or an
existing processed index in the project's Drive folder. The official dataset
release requires author-approved access. No multimodal accuracy is verified.

Validation: the new notebook passes nbformat validation and every Python cell
compiles. The focused preprocessing suite passes all 10 tests. The earlier full
review suite passed all 74 tests; this does not establish real-data model accuracy.

Downloader validation: three focused regression tests passed (403 fallback,
non-403 failure propagation, corrupt archive rejected without extraction).
