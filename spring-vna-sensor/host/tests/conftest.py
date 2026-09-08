import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import numpy as np

@pytest.fixture(scope='session')
def model():
    from honeycomb_host.fastmodel import FastModel, ModelConfig
    from honeycomb_host import geometry as G
    return FastModel(cfg=ModelConfig(gap=G.GAP_NOM))
