import shutil, os
shutil.rmtree(os.path.expanduser('~/.cache/torch/datasets'), ignore_errors=True)
shutil.rmtree(os.path.expanduser('~/.cache/torch/hub'), ignore_errors=True)