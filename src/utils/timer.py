import time
class Timer:
    def __enter__(self): self.t0=time.time(); return self
    def __exit__(self, exc_type, exc, tb): self.t1=time.time(); self.elapsed=self.t1-self.t0
