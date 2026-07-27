import logging
import sys
from collections import deque

log_buffer = deque(maxlen=200)

class BufferHandler(logging.Handler):
    def emit(self, record):
        log_buffer.append(self.format(record))

def setup_logger():
    logger = logging.getLogger("ai_shorts_factory")
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    
    buffer_handler = BufferHandler()
    buffer_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(handler)
        logger.addHandler(buffer_handler)
    
    return logger

logger = setup_logger()
