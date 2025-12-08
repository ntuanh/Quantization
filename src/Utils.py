

def get_size_frame(ret , frame):
    if ret :
        size_mb = frame.nbytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"