from hashlib import md5

SEP = '-'

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