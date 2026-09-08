#!/usr/bin/env python3
"""重生成 layout.json (与交接包同格式)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from honeycomb_host import geometry as G
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(G.DATA_DIR, 'layout.json')
G.write_layout(out); print('wrote', out)
