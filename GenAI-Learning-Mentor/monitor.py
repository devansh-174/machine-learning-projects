import csv
from datetime import datetime
import os
import time

import psutil

try:
    import pynvml

    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False


class PerformanceMonitor:

    def __init__(self):
        self.start_times = {}
        self.elapsed_times = {}
        self.csv_file = "performance_log.csv"

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "question",
                    "pdf_loading_time",
                    "chunking_time",
                    "embedding_time",
                    "generation_time",
                    "total_query",
                    "cpu_usage",
                    "ram_usage",
                    "gpu_utilization",
                    "gpu_memory_used",
                    "gpu_temperature",
                    "retrieved_chunks",
                    "context_length",
                    "input_tokens",
                    "output_tokens",
                    "embedding_model",
                    "llm",
                ])

    # ============================================
    # Reset all timers
    # ============================================
    def reset(self):
        self.start_times = {}
        self.elapsed_times = {}

    # ============================================
    # Start Timer
    # ============================================
    def start(self, name):
        self.start_times[name] = time.perf_counter()

    # ============================================
    # Stop Timer
    # ============================================
    def stop(self, name):
        if name not in self.start_times:
            return
        self.elapsed_times[name] = time.perf_counter() - self.start_times[name]

    # ============================================
    # CPU Usage
    # ============================================
    def get_cpu_usage(self):
        return psutil.cpu_percent(interval=0.2)

    # ============================================
    # RAM Usage
    # ============================================
    def get_ram_usage(self):
        return psutil.virtual_memory().percent

    # ============================================
    # GPU Information
    # ============================================
    def get_gpu_info(self):
        if not GPU_AVAILABLE:
            return {
                "gpu_name": "Not Available",
                "gpu_utilization": 0,
                "gpu_memory_used": 0,
                "gpu_memory_total": 0,
                "gpu_temperature": 0,
            }

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        temperature = pynvml.nvmlDeviceGetTemperature(
            handle, pynvml.NVML_TEMPERATURE_GPU
        )
        name = pynvml.nvmlDeviceGetName(handle)

        if isinstance(name, bytes):
            name = name.decode()

        return {
            "gpu_name": name,
            "gpu_utilization": utilization.gpu,
            "gpu_memory_used": round(memory.used / (1024**3), 2),
            "gpu_memory_total": round(memory.total / (1024**3), 2),
            "gpu_temperature": temperature,
        }

    # ============================================
    # Token Estimation
    # ============================================
    def estimate_tokens(self, text):
        if not text:
            return 0
        return int(len(text.split()) * 1.3)

    # ============================================
    # Pipeline Metrics
    # ============================================
    def collect_pipeline_metrics(self):
        return {
            "pdf_loading": round(self.elapsed_times.get("pdf_loading", 0), 3),
            "chunking": round(self.elapsed_times.get("chunking", 0), 3),
            "embedding_and_faiss": round(
                self.elapsed_times.get("embedding_and_faiss", 0), 3
            ),
            "pipeline": round(self.elapsed_times.get("pipeline", 0), 3),
        }

    # ============================================
    # Query Metrics
    # ============================================
    def collect_metrics(
        self,
        question,
        retrieved_chunks,
        context_length,
        answer,
        embedding_model,
        llm,
    ):
        gpu = self.get_gpu_info()

        # Handle context estimation dynamic payload safely
        if isinstance(context_length, str):
            context_tokens = self.estimate_tokens(context_length)
        elif isinstance(context_length, (int, float)):
            # If context_length is raw character length, 1 token ~ 4 chars is standard rule of thumb
            context_tokens = int(context_length / 4)
        else:
            context_tokens = 0

        metrics = {
            "question": question,
            "pdf_loading_time": round(self.elapsed_times.get("pdf_loading", 0), 3),
            "chunking_time": round(self.elapsed_times.get("chunking", 0), 3),
            "embedding_time": round(
                self.elapsed_times.get("embedding_and_faiss", 0), 3
            ),
            "generation_time": round(self.elapsed_times.get("generation", 0), 3),
            "total_query": round(self.elapsed_times.get("total_query", 0), 3),
            "cpu_usage": self.get_cpu_usage(),
            "ram_usage": self.get_ram_usage(),
            "gpu_name": gpu["gpu_name"],
            "gpu_utilization": gpu["gpu_utilization"],
            "gpu_memory_used": gpu["gpu_memory_used"],
            "gpu_memory_total": gpu["gpu_memory_total"],
            "gpu_temperature": gpu["gpu_temperature"],
            "retrieved_chunks": retrieved_chunks,
            "context_length": context_length,
            "estimated_input_tokens": self.estimate_tokens(question) + context_tokens,
            "estimated_output_tokens": self.estimate_tokens(answer),
            "embedding_model": embedding_model,
            "llm": llm,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics["timestamp"],
                metrics["question"],
                metrics["pdf_loading_time"],
                metrics["chunking_time"],
                metrics["embedding_time"],
                metrics["generation_time"],
                metrics["total_query"],
                metrics["cpu_usage"],
                metrics["ram_usage"],
                metrics["gpu_utilization"],
                metrics["gpu_memory_used"],
                metrics["gpu_temperature"],
                metrics["retrieved_chunks"],
                metrics["context_length"],
                metrics["estimated_input_tokens"],
                metrics["estimated_output_tokens"],
                metrics["embedding_model"],
                metrics["llm"],
            ])

        return metrics