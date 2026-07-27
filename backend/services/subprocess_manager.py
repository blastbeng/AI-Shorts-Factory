import subprocess
from backend.services.logger import logger

class SubprocessManager:
    _processes = []

    @staticmethod
    def add(proc: subprocess.Popen):
        SubprocessManager._processes.append(proc)

    @staticmethod
    def remove(proc: subprocess.Popen):
        if proc in SubprocessManager._processes:
            SubprocessManager._processes.remove(proc)

    @staticmethod
    def kill_all():
        logger.info("Terminazione di tutti i processi AI esterni in corso...")
        for proc in SubprocessManager._processes:
            if proc.poll() is None:  # Se il processo è ancora in esecuzione
                logger.warning(f"Uccisione processo forzata: {proc.args}")
                proc.kill()
        SubprocessManager._processes.clear()
