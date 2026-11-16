"""Isolate eSpeak's native data-path limitation from long project/environment paths."""
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace


class EnglishFrontend:
    def __init__(self):
        import espeakng_loader
        from misaki.espeak import EspeakFallback
        from phonemizer.backend.espeak.wrapper import EspeakWrapper

        # The same package data fails under this project's longer .venv path but
        # succeeds in a short directory. Keep a private copy for the backend lifetime.
        self._temporary = tempfile.TemporaryDirectory(prefix="pv-espeak-")
        target = Path(self._temporary.name) / "espeak-ng-data"
        shutil.copytree(espeakng_loader.get_data_path(), target)
        EspeakWrapper.set_data_path(str(target))
        self._frontend = EspeakFallback(british=False)

    def __call__(self, text: str) -> str:
        phonemes, _ = self._frontend(SimpleNamespace(text=text))
        return phonemes
