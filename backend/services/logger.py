import logging
import sys
import os
from collections import deque

log_buffer = deque(maxlen=200)
LOG_FILE_PATH = "logs/app.log"

class BufferHandler(logging.Handler):
    def emit(self, record):
        log_buffer.append(self.format(record))

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

class TqdmToLogger:
    def __init__(self, logger, level=logging.INFO):
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, buf):
        self.buffer += buf
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            # Handle carriage returns from tqdm by keeping only the last update
            if '\r' in line:
                line = line.split('\r')[-1]
            if line.strip():
                self.logger.log(self.level, line.strip())

    def flush(self):
        if self.buffer.strip():
            line = self.buffer.split('\r')[-1]
            if line.strip():
                self.logger.log(self.level, line.strip())
        self.buffer = ""

def load_existing_logs():
    if os.path.exists(LOG_FILE_PATH):
        with open(LOG_FILE_PATH, "r") as f:
            lines = f.readlines()
            # Load the last 200 lines into the buffer
            for line in lines[-200:]:
                log_buffer.append(line.strip())

def setup_logger():
    logger = logging.getLogger("ai_shorts_factory")
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load existing logs into buffer on startup
    load_existing_logs()
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)
    
    if not logger.handlers:
        # Console handler
        handler = FlushStreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Buffer handler for API
        buffer_handler = BufferHandler()
        buffer_handler.setFormatter(formatter)
        logger.addHandler(buffer_handler)
        
        # File handler for persistence
        file_handler = logging.FileHandler(LOG_FILE_PATH)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Redirect stderr to logger to capture tqdm progress bars
    sys.stderr = TqdmToLogger(logger)
    
    return logger

logger = setup_logger()
