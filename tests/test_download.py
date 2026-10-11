import hashlib

import pytest
import requests

from deepfake.data import download


@pytest.mark.parametrize("status", [403, 500])
def test_asvspoof_fallback_only_for_forbidden(tmp_path, monkeypatch, status):
    archive = tmp_path / "LA.zip"
    payload = b"verified test archive"
    monkeypatch.setattr(download, "ASVSPOOF_LA_MD5", hashlib.md5(payload).hexdigest())
    calls = []
    extracted = []

    def fetch(url, dest, expected_size):
        calls.append(url)
        if url == download.ASVSPOOF_LA_URL:
            response = requests.Response()
            response.status_code = status
            raise requests.HTTPError(response=response)
        dest.write_bytes(payload)
        return dest

    monkeypatch.setattr(download, "http_download", fetch)
    monkeypatch.setattr(download, "extract", lambda *args: extracted.append(args))
    if status == 500:
        with pytest.raises(requests.HTTPError):
            download.get_asvspoof(tmp_path)
        assert calls == [download.ASVSPOOF_LA_URL]
        assert not extracted
    else:
        download.get_asvspoof(tmp_path, keep_zip=True)
        assert calls == [download.ASVSPOOF_LA_URL, download.ASVSPOOF_LA_MIRROR]
        assert extracted == [(archive, tmp_path)]
        assert archive.read_bytes() == payload


def test_asvspoof_corrupt_archive_never_extracted(tmp_path, monkeypatch):
    archive = tmp_path / "LA.zip"
    archive.write_bytes(b"corrupt archive")
    monkeypatch.setattr(download, "http_download", lambda *args, **kwargs: archive)
    monkeypatch.setattr(download, "extract", lambda *args: pytest.fail("Corrupt archive extracted"))
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download.get_asvspoof(tmp_path)
    assert archive.exists()
