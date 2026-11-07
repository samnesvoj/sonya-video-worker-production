"""python -m modes.trailer_cinematic --input runtime/dl/kishlak.mp4 --duration 60      (v2 recap, default)
   python -m modes.trailer_cinematic --v1 --input ...                                  (legacy v1 trailer)"""
import sys

if "--v1" in sys.argv:
    from .runner import main
    sys.exit(main([a for a in sys.argv[1:] if a != "--v1"]))
from .recap_runner import main

sys.exit(main())
