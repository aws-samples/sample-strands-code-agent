"""Tests for strands_code_agent.utils — image_to_base64."""

import base64
import os
import tempfile

from strands_code_agent.utils import image_to_base64


class TestImageToBase64:
    def test_roundtrip(self):
        raw = b"\x89PNG\r\n\x1a\nfake-image-data"
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            result = image_to_base64(path)
            assert isinstance(result, str)
            assert base64.b64decode(result) == raw
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            path = f.name
        try:
            result = image_to_base64(path)
            assert result == ""
        finally:
            os.unlink(path)

    def test_binary_content(self):
        raw = bytes(range(256))
        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            f.write(raw)
            path = f.name
        try:
            result = image_to_base64(path)
            assert base64.b64decode(result) == raw
        finally:
            os.unlink(path)
