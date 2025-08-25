import ray
import torch
import torch.distributed as dist
import socket
import time
from typing import List

@ray.remote
class Coordinator:
    def __init__(self, total_processes: int):
        self.total_processes = total_processes
        self.registrations = []  # store (ip, port) of every process
        self.ready = False

    def register(self, ip: str, port: int):
        self.registrations.append((ip, port))
        current_size = len(self.registrations)
        if current_size == self.total_processes:
            self.ready = True
        return current_size - 1  # return rank

    def get_addresses(self):
        while not self.ready:
            time.sleep(0.1)
        return self.registrations