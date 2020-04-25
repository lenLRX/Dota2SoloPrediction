import os
import re


def get_latest_params(path):
    files = os.listdir(path)
    latest_n = None
    latest_fname = None
    for fname in files:
        r = re.search("SoloPred_(\\d+)\\.model", fname)
        if r:
            n = int(r.group(1))
            if not latest_n or n > latest_n:
                latest_n = n
                latest_fname = fname
    assert latest_fname is not None
    return latest_fname


if __name__ == "__main__":
    print(get_latest_params("./save"))
