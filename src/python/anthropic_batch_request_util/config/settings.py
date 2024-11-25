from typing import List, Optional, Dict, Any
from pathlib import Path
import os
import yaml
from pydantic import BaseModel, Field, validator, root_validator
import logging

logger = logging.getLogger(__name__)


class APIConfig(BaseModel):
    """API related configuration"""

    version: str
    beta_features: List[str]
    timeout: int
    max_payload_size: int = Field(default=104857600)  # 100MB

    @validator("max_payload_size")
    def validate_payload_size(cls, v):
        if v <= 0 or v > 104857600:  # 100MB limit
            raise ValueError("Payload size must be between 0 and 100MB")
        return v


class ModelConfig(BaseModel):
    """Model specific configuration"""

    name: str
    display_name: str
    max_tokens: int
    temperature: float
    stop_sequences: Optional[List[str]] = None
    response_format: str = "text"

    @validator("temperature")
    def validate_temperature(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Temperature must be between 0 and 1")
        return v

    @validator("response_format")
    def validate_response_format(cls, v):
        if v not in ["text", "json"]:
            raise ValueError("response_format must be either 'text' or 'json'")
        return v


class BatchConfig(BaseModel):
    """Batch processing configuration"""

    max_size: int
    chunk_size: int
    enable_prompt_cache: bool
    cache_type: str = "ephemeral"

    @validator("chunk_size")
    def validate_chunk_size(cls, v, values):
        if "max_size" in values and v > values["max_size"]:
            raise ValueError("chunk_size cannot be larger than max_size")
        return v


class StorageConfig(BaseModel):
    """Storage related configuration"""

    base_dir: Path
    subdirs: Dict[str, str]
    request_file_prefix: str
    result_file_prefix: str

    @validator("base_dir", pre=True)
    def validate_base_dir(cls, v):
        return Path(v)

    def get_path(self, subdir: str) -> Path:
        """Get full path for a specific subdirectory"""
        if subdir not in self.subdirs:
            raise ValueError(f"Unknown subdirectory: {subdir}")
        return self.base_dir / self.subdirs[subdir]


class WorkerBrokerConfig(BaseModel):
    """Celery broker configuration"""

    url: str
    connection_retry: bool
    connection_retry_on_startup: bool


class WorkerBackendConfig(BaseModel):
    """Celery result backend configuration"""

    url: str
    result_expires: Optional[int] = None


class WorkerTaskConfig(BaseModel):
    """Celery task configuration"""

    serializer: str
    result_serializer: str
    accept_content: List[str]
    default_queue: str
    track_started: bool
    ignore_result: bool
    create_missing_queues: bool
    delivery_mode: str
    always_eager: bool


class WorkerConfig(BaseModel):
    """Worker specific configuration"""

    prefetch_multiplier: int
    max_tasks_per_child: int
    enable_utc: bool
    timezone: str


class MonitorConfig(BaseModel):
    """Monitor task configuration"""

    retry_delay: int
    max_retries: int
    initial_delay: int


class BeatConfig(BaseModel):
    """Celery beat configuration"""

    schedule_filename: str


class BaseTaskConfig(BaseModel):
    """Base configuration for all tasks"""

    track_started: bool = True
    ignore_result: bool = False
    acks_late: bool = True
    reject_on_worker_lost: bool = True
    retry_backoff: bool = True
    retry_jitter: bool = True
    retry_backoff_max: int = 600
    time_limit: Optional[int] = None
    soft_time_limit: Optional[int] = None


class TaskConfig(BaseTaskConfig):
    """Individual task configuration extending base config"""

    name: str
    max_retries: Optional[int]
    retry_delay: int
    queue: str
    eta_delay: int

    @root_validator(pre=True)
    def merge_with_base_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Merge task specific config with base config"""
        base_config = {
            "track_started": True,
            "ignore_result": False,
            "acks_late": True,
            "reject_on_worker_lost": True,
            "retry_backoff": True,
            "retry_jitter": True,
            "retry_backoff_max": 600,
            "time_limit": None,
            "soft_time_limit": None,
        }
        return {**base_config, **values}


class TasksConfig(BaseModel):
    """Task specific configurations"""

    class MonitorBatchConfig(TaskConfig):
        """Monitor batch task specific configuration"""

        name: str = "anthropic_batch_tasks.monitor_batch"
        max_retries: Optional[int] = None  # null for infinite retries
        retry_delay: int = 60
        queue: str = "anthropic"
        eta_delay: int = 30

    monitor_batch: MonitorBatchConfig = MonitorBatchConfig()


class CeleryConfig(BaseModel):
    """Complete Celery configuration"""

    broker: WorkerBrokerConfig
    backend: WorkerBackendConfig
    task: WorkerTaskConfig
    worker: WorkerConfig
    monitor: MonitorConfig
    beat: BeatConfig
    base_task: BaseTaskConfig
    tasks: TasksConfig


class Settings(BaseModel):
    """Main settings class that combines all configurations"""

    class Config:
        env_prefix = "ANTHROPIC_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    api: APIConfig
    model: ModelConfig
    batch: BatchConfig
    storage: StorageConfig
    worker: CeleryConfig

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Settings":
        """Load settings from YAML file and environment variables"""
        if config_path is None:
            config_path = os.getenv("ANTHROPIC_CONFIG_PATH", "config/settings.yaml")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)

            # Create base directories
            settings = cls(**config_data)
            for subdir in settings.storage.subdirs.values():
                path = settings.storage.base_dir / subdir
                path.mkdir(parents=True, exist_ok=True)

            return settings

        except Exception as e:
            logger.error(f"Failed to load settings from {config_path}: {e}")
            raise

    def get_celery_config(self) -> Dict[str, Any]:
        """Get Celery configuration dictionary"""
        return {
            "broker_url": self.worker.broker.url,
            "result_backend": self.worker.backend.url,
            "task_serializer": self.worker.task.serializer,
            "result_serializer": self.worker.task.result_serializer,
            "accept_content": self.worker.task.accept_content,
            "task_default_queue": self.worker.task.default_queue,
            "task_track_started": self.worker.task.track_started,
            "task_ignore_result": self.worker.task.ignore_result,
            "result_expires": self.worker.backend.result_expires,
            "worker_prefetch_multiplier": self.worker.worker.prefetch_multiplier,
            "worker_max_tasks_per_child": self.worker.worker.max_tasks_per_child,
            "beat_schedule_filename": self.worker.beat.schedule_filename,
            "timezone": self.worker.worker.timezone,
            "enable_utc": self.worker.worker.enable_utc,
            "task_always_eager": self.worker.task.always_eager,
            "task_create_missing_queues": self.worker.task.create_missing_queues,
            "task_default_delivery_mode": self.worker.task.delivery_mode,
            "broker_connection_retry": self.worker.broker.connection_retry,
            "broker_connection_retry_on_startup": self.worker.broker.connection_retry_on_startup,
        }


# Create global settings instance
settings = Settings.load()
