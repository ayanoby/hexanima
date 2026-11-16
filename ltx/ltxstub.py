"""import ltx_core submodules without running the package __init__ files (they import transformers Gemma3 classes the venv does not have)"""
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sys, types, os
R = (os.environ.get('LTX_C', 'ltxc'))
for name in ('ltx_core', 'ltx_core.model', 'ltx_core.model.transformer', 'ltx_core.text_encoders', 'ltx_core.text_encoders.gemma', 'ltx_core.loader'):
    m = types.ModuleType(name); m.__path__ = [os.path.join(R, *name.split('.'))]; sys.modules[name] = m
