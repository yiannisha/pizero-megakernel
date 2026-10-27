"""Development CUPTI driver callback trace; use separately from Kineto/sanitizer."""
import ctypes


class LaunchTrace:
    def __init__(self, library):
        self.library=ctypes.CDLL(str(library))
        self.begin=self.library.begin_launch_trace
        self.begin.argtypes=[ctypes.POINTER(ctypes.c_void_p)]
        self.begin.restype=ctypes.c_int
        self.end=self.library.end_launch_trace
        self.end.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int),ctypes.c_void_p,ctypes.c_int]
        self.end.restype=ctypes.c_int
        self.error=self.library.launch_trace_error
        self.error.argtypes=[ctypes.c_int]
        self.error.restype=ctypes.c_char_p
        self.handle=ctypes.c_void_p()

    def check(self, code):
        if code:
            raise RuntimeError(f'CUPTI launch trace error {code}: {self.error(code)}')

    def __enter__(self):
        self.check(self.begin(ctypes.byref(self.handle)))
        return self

    def __exit__(self, *_):
        count=ctypes.c_int()
        symbol=ctypes.create_string_buffer(2048)
        code=self.end(self.handle,ctypes.byref(count),symbol,len(symbol))
        self.handle=ctypes.c_void_p()
        self.check(code)
        self.count=count.value
        self.symbol=symbol.value.decode()
