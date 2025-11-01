import shutil
import time

import os

def remove_cached_files():
    
    while True:
        shutil.rmtree('./data', ignore_errors=True)
        os.makedirs("./data")
        time.sleep(999)


remove_cached_files()
