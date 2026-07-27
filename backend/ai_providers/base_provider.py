from abc import ABC, abstractmethod

class BaseAIProvider(ABC):
    @abstractmethod
    def install_status(self):
        pass

    @abstractmethod
    def health_check(self):
        pass

    @abstractmethod
    def generate(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_capabilities(self):
        pass

    @abstractmethod
    def get_gpu_requirements(self):
        pass
