from hashlib import md5
import os
from contextlib import contextmanager
from tempfile import TemporaryDirectory

SEP = '-'

@contextmanager
def atomic_output(path):
    """Publish an output file after its writer has closed it successfully."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with TemporaryDirectory(prefix=".enerzyme-", dir=parent) as staging:
        temporary = os.path.join(staging, os.path.basename(path))
        yield temporary
        os.replace(temporary, path)

def hash_model_name(model_name, params, idx=8):
    model_str = ''
    for key, value in sorted(params.items()):
        if key == 'active':
            continue
        model_str += str(key) + str(value)
    model_str = model_name + SEP + md5(model_str.encode('utf-8')).hexdigest()[:idx]
    return model_str

def model_name_generation(model_id, model_name, feature_names, task, joiner=SEP):
    return joiner.join([model_id, model_name, '_'.join(sorted(feature_names)), task])
