# FAQ

## Question
If users met this error on Mac:
```
Traceback (most recent call last):
  File "/Users/bytedance/Downloads/music_source_separation/separate_scripts/separate.py", line 7, in <module>
    from bytesep.inference import SeparatorWrapper
  File "/Users/bytedance/miniconda3/envs/tmp/lib/python3.9/site-packages/bytesep-0.0.1-py3.9.egg/bytesep/__init__.py", line 1, in <module>
    from bytesep.inference import Separator
  File "/Users/bytedance/miniconda3/envs/tmp/lib/python3.9/site-packages/bytesep-0.0.1-py3.9.egg/bytesep/inference.py", line 13, in <module>
    from bytesep.models.lightning_modules import get_model_class
ModuleNotFoundError: No module named 'bytesep.models'
```

Solution

Add environment path by:
```
$ PYTHONPATH="./"
$ export PYTHONPATH
```

## Question

If users met this error:
```
Segmentation fault: 11
```

Solution

Try to use python 3.7. This error sometimes occur when using python 3.9.

## Question

If users met this error:

```
horovod.common.exceptions.HorovodVersionMismatchError:
```

Solution

```
pip uninstall horovod
```