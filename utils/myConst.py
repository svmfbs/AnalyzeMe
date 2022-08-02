""" Constant types in Python """
import sys
sys.dont_write_bytecode = True # dont make .pyc files

class _const:
    """ constant class """
    class ConstError(TypeError):
        pass

    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise self.ConstError(f"Cannot rebind const ({name})")
        self.__dict__[name] = value

sys.modules[__name__] = _const()
