"""A small retry for the package's network calls.

IRSA TAP/SIA, S3 byte-range reads and unwise.me downloads sometimes fail
transiently, so every network call goes through ``retry``: a few tries
with a short wait in between.
"""
import time

# OSError covers socket/timeout errors (s3fs, urllib); pyvo/astroquery raise
# requests exceptions, which subclass OSError in requests>=2.
_DEFAULT_EXCEPTIONS = (OSError, TimeoutError)

# anon=True is required to read the public IRSA S3 bucket.
S3_FSSPEC_KWARGS = {"anon": True}


def retry(fn, *, attempts=3, base_delay=2.0,
          exceptions=_DEFAULT_EXCEPTIONS, what=""):
    """Call ``fn()`` up to ``attempts`` times, waiting between tries.

    ``fn`` is a zero-argument callable (close over its arguments). Only
    ``exceptions`` are retried; a ``ValueError`` (bad request, missing
    product) fails immediately. The last failure re-raises.
    """
    for k in range(attempts):
        try:
            return fn()
        except exceptions as exc:
            if k == attempts - 1:
                raise
            print(f"  {what or 'network call'} failed ({exc!r}); "
                  f"retry {k + 1}/{attempts - 1} in {base_delay:.0f}s")
            time.sleep(base_delay)
