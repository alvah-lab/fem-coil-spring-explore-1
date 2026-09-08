#!/usr/bin/env python3
"""生成测试 fixture (唯一允许导入旧 FastHenry 脚本之处; 手动运行, 需 FastHenry).
- fixtures/series_scheme_T.npz: 旧 series_scheme 的 T_phys/T_s/RANGE_q (导入会跑一次 FastHenry, ~1s)
"""
import sys, os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); HOST = os.path.dirname(HERE)
FH = os.path.join(os.path.dirname(HOST), 'fasthenry_runs', 'honeycomb')
sys.path.insert(0, FH); os.chdir(FH)
import series_scheme as S     # noqa: 导入即跑 FastHenry
os.makedirs(os.path.join(HOST, 'tests', 'fixtures'), exist_ok=True)
np.savez(os.path.join(HOST, 'tests', 'fixtures', 'series_scheme_T.npz'), T_phys=S.T_phys, T_s=S.T_s, RANGE_q=S.RANGE_q,
         units=np.array(S.UNITS), edges=np.array(S.EDGES))
print('saved fixtures/series_scheme_T.npz')
