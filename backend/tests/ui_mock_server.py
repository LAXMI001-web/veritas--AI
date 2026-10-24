"""TEST-ONLY server: real audio model + MOCKED FA3 scores (suspicious burst in the middle) so the UI can be exercised
without the 580 MB checkpoint. Never use for real analysis. Run: python -m tests.ui_mock_server"""
import os, sys, tempfile
os.environ.setdefault("VERITAS_DATA_DIR", tempfile.mkdtemp(prefix="veritas_ui_"))
from tests.conftest import MockFA3
from app import registry
registry.set_detectors(fa3=MockFA3(lambda i: 0.93 if 6 <= i % 16 <= 9 else 0.08))
import uvicorn
from app.main import app
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]) if len(sys.argv) > 1 else 8766)
