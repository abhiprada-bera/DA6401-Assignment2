import sys
from types import ModuleType

class MockMod(ModuleType):
    def __getattr__(self, name):
        if name == "Compose":
            return lambda *a, **kw: None
        if name == "pytorch":
             m = MockMod("pytorch")
             m.ToTensorV2 = lambda *a, **kw: None
             return m
        return lambda *a, **kw: None

m = MockMod("albumentations")
m.pytorch = MockMod("pytorch")
m.pytorch.ToTensorV2 = lambda *a,**kw: None
sys.modules["albumentations"] = m
sys.modules["albumentations.pytorch"] = m.pytorch

import assignment2_part4
