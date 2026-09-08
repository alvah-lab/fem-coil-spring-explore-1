import os, sys, pytest
pytest.importorskip('PyQt6'); pytest.importorskip('pyqtgraph')

def test_gui_offscreen(tmp_path):
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    from honeycomb_host.gui.main import main
    shot = str(tmp_path / 'shot.png')
    rc = main(['--source', 'twin', '--scene', 'rest', '--noise', 'off', '--screenshot', shot, '--seconds', '1.0'])
    assert rc == 0 and os.path.exists(shot)
