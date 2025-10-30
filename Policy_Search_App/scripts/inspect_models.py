import joblib, os
from pprint import pprint

models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
print('Found model files:', files)
for fname in files:
    path = os.path.join(models_dir, fname)
    try:
        obj = joblib.load(path)
    except Exception as e:
        print('\nFAILED loading', fname, e)
        continue
    print('\nFile:', fname)
    print(' Type:', type(obj))
    if hasattr(obj, 'shape'):
        print(' Shape:', getattr(obj, 'shape'))
    if isinstance(obj, dict):
        print(' Keys:', list(obj.keys())[:10])
        # print small sample of dict values types
        for k in list(obj.keys())[:10]:
            v = obj[k]
            print('  -', k, '->', type(v), getattr(v, 'shape', None))
    elif isinstance(obj, (list, tuple)):
        print(' Length:', len(obj))
        print(' Sample types:', [type(x) for x in obj[:5]])
    else:
        try:
            # attempt to show a small slice if numpy-like
            import numpy as np
            arr = np.asarray(obj)
            print(' Asarray shape:', arr.shape, 'dtype:', arr.dtype)
            if arr.size and arr.dtype == object:
                print('  Object sample:', arr.flatten()[:5])
        except Exception as e:
            print('  Could not jsonify sample:', e)
print('\nDone')
