import os
import json
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime, timedelta
from celery import Celery
import pytz
from enum import Enum, auto
from dataclasses import dataclass

from .batch_handler import (
    AnthropicBatchHandler,
    BatchProcessingError,
    BatchStatus,
    AnthropicErrorType,
)
from .config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Celery with settings
celery_app = Celery("anthropic_batch_tasks")
celery_app.conf.update(settings.get_celery_config())


class TaskErrorType(Enum):
    """Task specific error types for monitoring"""

    TEMPORARY = auto()  # Temporary errors that can be retried
    PERMANENT = auto()  # Permanent errors that should not be retried
    UNKNOWN = auto()  # Unknown errors that need investigation and special logigng


class TaskError(Exception):
    """Custom exception for task-specific errors"""

    def __init__(
        self,
        message: str,
        error_type: TaskErrorType = TaskErrorType.UNKNOWN,
        original_error: Exception | None = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error


class ValidationError(BatchProcessingError):
    """Exception for validation errors"""

    def __init__(self, message: str):
        super().__init__(message, error_type=AnthropicErrorType.INVALID_REQUEST)


def _classify_task_error(error: Exception) -> TaskErrorType:
    """Classify task-specific errors for retry decisions"""
    if isinstance(error, TaskError):
        return error.error_type

    if isinstance(error, BatchProcessingError):
        # Map Anthropic error types to task error types
        anthropic_to_task = {
            AnthropicErrorType.RATE_LIMIT: TaskErrorType.TEMPORARY,
            AnthropicErrorType.SERVICE_UNAVAILABLE: TaskErrorType.TEMPORARY,
            AnthropicErrorType.SERVER_ERROR: TaskErrorType.TEMPORARY,
            AnthropicErrorType.INVALID_REQUEST: TaskErrorType.PERMANENT,
            AnthropicErrorType.AUTHENTICATION: TaskErrorType.PERMANENT,
            AnthropicErrorType.PERMISSION: TaskErrorType.PERMANENT,
        }
        return anthropic_to_task.get(error.error_type, TaskErrorType.UNKNOWN)

    # Celery specific errors
    if any(
        err_type in str(error.__class__)
        for err_type in ["ConnectionError", "OperationalError", "TimeoutError"]
    ):
        return TaskErrorType.TEMPORARY

    return TaskErrorType.UNKNOWN


def get_monitor_task_config() -> Dict[str, Any]:
    """
    Get monitor task configuration from settings.

    Returns:
        Dictionary containing task configuration
    """
    task_settings = settings.worker.tasks.monitor_batch
    return task_settings.dict(exclude_none=True)


@celery_app.task(**get_monitor_task_config())
def monitor_batch(
    job_id: str,
    result_path: str,
    start_time: str,
    scheduled_time: str | None = None,
) -> Dict[str, Any]:
    """
    Monitor batch job status and handle results.

    Args:
        job_id: Batch job ID to monitor
        result_path: Path where results should be saved
        start_time: ISO format timestamp when the batch job started
        scheduled_time: Optional scheduled time for this monitoring task

    Returns:
        Dict containing monitoring status and results
    """
    timezone = pytz.timezone(settings.worker.worker.timezone)
    current_time = datetime.now(timezone)

    logger.info(f"Starting monitoring task for job_id: {job_id}")
    logger.info(f"Current time: {current_time.isoformat()}")

    try:
        handler = AnthropicBatchHandler()
        status, result = _handle_batch_status(
            handler, job_id, result_path, start_time, current_time
        )
        return result
    except Exception as e:
        return _handle_monitoring_error(
            e, job_id, result_path, start_time, current_time
        )


def _handle_batch_status(
    handler: AnthropicBatchHandler,
    job_id: str,
    result_path: str,
    start_time: str,
    current_time: datetime,
) -> Tuple[BatchStatus, Dict[str, Any]]:
    """
    Handle different batch processing statuses.
    """
    try:
        # Get status and metrics
        raw_status, results, _ = handler.check_batch_status(job_id)
        status = BatchStatus.from_str(raw_status)
        metrics = handler.get_batch_metrics(job_id)

        # Common metadata for all status responses
        base_metadata = {
            "job_id": job_id,
            "status": status.value,
            "current_time": current_time.isoformat(),
            "metrics": metrics,
            "elapsed_time": (
                current_time - datetime.fromisoformat(start_time)
            ).total_seconds(),
        }

        if status == BatchStatus.ENDED:
            # Serialize results before saving
            serialized_results = handler.serialize_batch_results(results)

            # Add completion metadata
            metadata = {
                **base_metadata,
                "completion_time": current_time.isoformat(),
                "final_status": True,
            }

            # Determine final status based on metrics
            if metrics["request_counts"]["errored"] > 0:
                metadata["warning"] = "Batch completed with errors"
                metadata["status"] = "completed_with_errors"
            elif metrics["request_counts"]["expired"] > 0:
                metadata["warning"] = "Batch completed with expired requests"
                metadata["status"] = "completed_with_expired"
            else:
                metadata["status"] = "completed_successfully"

            return status, _save_batch_results(
                serialized_results, result_path, metadata
            )

        elif status == BatchStatus.PROCESSING:
            monitoring_result = _schedule_next_monitoring(
                monitor_batch, job_id, result_path, start_time
            )
            return status, {**base_metadata, **monitoring_result}

        elif status == BatchStatus.CANCELING:
            return status, {
                **base_metadata,
                "message": "Batch is being canceled",
                "cancellation_detected_at": current_time.isoformat(),
            }

        else:  # BatchStatus.UNKNOWN or unexpected status
            logger.warning(f"Unexpected batch status received: {raw_status}")
            return BatchStatus.UNKNOWN, {
                **base_metadata,
                "warning": f"Unexpected batch status: {raw_status}",
                "original_status": raw_status,
            }

    except BatchProcessingError as e:
        # Log the error with its specific type
        logger.error(
            f"Batch processing error for job {job_id}: "
            f"{str(e)} (type: {e.error_type.name})"
        )
        raise
    except Exception as e:
        # Wrap unknown errors in BatchProcessingError
        error_type = AnthropicErrorType.from_error(e)
        raise BatchProcessingError(
            f"Status check failed: {str(e)}", error_type=error_type, original_error=e
        )


def _save_batch_results(
    results: List[Dict[str, Any]], result_path: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save batch results to file with enhanced error handling and validation.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(result_path), exist_ok=True)

        # Validate results structure
        for result in results:
            if not isinstance(result, dict) or "custom_id" not in result:
                raise ValidationError("Invalid result format detected")

        # Prepare result data with timestamp
        result_data = {
            **metadata,
            "save_timestamp": datetime.now(pytz.UTC).isoformat(),
            "results": results,
            "result_count": len(results),
        }

        # Save with atomic write operation
        temp_path = f"{result_path}.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, result_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        logger.info(f"Successfully saved {len(results)} results to {result_path}")

        return {
            **metadata,
            "status": metadata.get("status", "completed"),
            "result_path": result_path,
            "result_count": len(results),
        }

    except ValidationError as e:
        logger.error(f"Validation error while saving results: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to save results: {str(e)}")
        raise BatchProcessingError(
            f"Failed to save results: {str(e)}",
            error_type=AnthropicErrorType.UNKNOWN,
            original_error=e,
        )


def _handle_monitoring_error(
    error: Exception,
    job_id: str,
    result_path: str,
    start_time: str,
    current_time: datetime,
) -> Dict[str, Any]:
    """Enhanced error handling with retry strategy based on error type"""
    error_type = _classify_task_error(error)
    logger.error(
        f"Monitoring error for job {job_id}: {str(error)} (type: {error_type.name})",
        exc_info=True,
    )

    try:
        if error_type == TaskErrorType.PERMANENT:
            return {
                "status": "failed",
                "job_id": job_id,
                "error": str(error),
                "error_type": error_type.name,
                "message": "Permanent error - no retry scheduled",
            }

        # Calculate retry delay with exponential backoff
        base_delay = settings.worker.tasks.monitor_batch.retry_delay
        max_delay = settings.worker.tasks.monitor_batch.retry_backoff_max
        retry_count = getattr(monitor_batch.request, "retries", 0)

        delay = min(base_delay * (2**retry_count), max_delay)

        # Add jitter if configured
        if settings.worker.tasks.monitor_batch.retry_jitter:
            import random

            jitter = random.uniform(0.8, 1.2)
            delay = int(delay * jitter)

        # Schedule next attempt
        monitoring_result = _schedule_next_monitoring(
            monitor_batch, job_id, result_path, start_time, delay=delay
        )

        monitoring_result.update(
            {
                "error_type": error_type.name,
                "original_error": str(error),
                "retry_count": retry_count + 1,
                "retry_delay": delay,
            }
        )

        return monitoring_result

    except Exception as schedule_error:
        logger.error(
            f"Failed to handle monitoring error: {str(schedule_error)}", exc_info=True
        )
        return {
            "status": "error",
            "job_id": job_id,
            "error": f"Error handling failed: {str(schedule_error)}",
            "original_error": str(error),
            "error_type": error_type.name,
        }


@dataclass
class MonitoringMetrics:
    """Monitoring metrics data structure"""

    monitoring_count: int
    total_duration: float
    average_interval: float
    last_check_time: str
    status_history: List[str]
    error_count: int
    retry_count: int


class MonitoringState:
    """Monitoring state management"""

    def __init__(self, task_instance: Any):
        self.task = task_instance
        self.headers = getattr(task_instance.request, "headers", {}) or {}
        self.task_metadata = self.headers.get("task_metadata", {})

    @property
    def monitoring_count(self) -> int:
        return self.task_metadata.get("monitoring_count", 0)

    @property
    def retry_count(self) -> int:
        return getattr(self.task.request, "retries", 0)

    def get_monitoring_metrics(
        self, start_time: str, current_time: datetime
    ) -> MonitoringMetrics:
        """Get current monitoring metrics"""
        history = self.task_metadata.get("status_history", [])
        error_count = self.task_metadata.get("error_count", 0)

        total_duration = (
            current_time - datetime.fromisoformat(start_time)
        ).total_seconds()

        avg_interval = (
            total_duration / self.monitoring_count if self.monitoring_count > 0 else 0
        )

        return MonitoringMetrics(
            monitoring_count=self.monitoring_count,
            total_duration=total_duration,
            average_interval=avg_interval,
            last_check_time=current_time.isoformat(),
            status_history=history,
            error_count=error_count,
            retry_count=self.retry_count,
        )


def _update_monitoring_state(
    task: Any, job_id: str, current_status: str, error: Exception | None = None
) -> Dict[str, Any]:
    """Update monitoring state with current status"""
    if not hasattr(task, "request"):
        task_metadata = {}
    else:
        headers = getattr(task.request, "headers", {}) or {}
        task_metadata = headers.get("task_metadata", {}).copy()

    # Update status history
    status_history = task_metadata.get("status_history", [])
    status_history.append(
        {
            "status": current_status,
            "timestamp": datetime.now(pytz.UTC).isoformat(),
            "error": str(error) if error else None,
        }
    )

    # Update error count
    error_count = task_metadata.get("error_count", 0)
    if error:
        error_count += 1

    # Prepare updated metadata
    updated_metadata = {
        **task_metadata,
        "monitoring_count": task_metadata.get("monitoring_count", 0) + 1,
        "status_history": status_history[-10:],  # Keep last 10 statuses
        "error_count": error_count,
        "last_status": current_status,
    }

    return updated_metadata


def _schedule_next_monitoring(
    task: Any,
    job_id: str,
    result_path: str,
    start_time: str,
    delay: int | None = None,
    current_status: str = "in_progress",
    error: Exception | None = None,
) -> Dict[str, Any]:
    """
    Enhanced scheduling with state management and metrics.
    """
    timezone = pytz.timezone(settings.worker.worker.timezone)
    current_time = datetime.now(timezone)
    eta_delay = delay or settings.worker.tasks.monitor_batch.eta_delay
    next_check_time = current_time + timedelta(seconds=eta_delay)

    try:
        # Update monitoring state
        task_metadata = _update_monitoring_state(task, job_id, current_status, error)

        # Get monitoring metrics
        state = MonitoringState(task)
        metrics = state.get_monitoring_metrics(start_time, current_time)

        # Check monitoring limits
        if (
            hasattr(settings.worker.monitor, "max_monitoring_time")
            and metrics.total_duration > settings.worker.monitor.max_monitoring_time
        ):
            raise TaskError(
                "Maximum monitoring time exceeded", error_type=TaskErrorType.PERMANENT
            )

        # Schedule next task with updated metadata
        next_task = task.apply_async(
            args=[job_id, result_path],
            kwargs={
                "start_time": start_time,
                "scheduled_time": next_check_time.isoformat(),
            },
            eta=next_check_time,
            headers={"task_metadata": task_metadata},
        )

        logger.info(
            f"Scheduled next monitoring task {next_task.id} "
            f"for job {job_id} at {next_check_time.isoformat()} "
            f"(monitoring count: {metrics.monitoring_count})"
        )

        return {
            "status": "scheduled",
            "job_id": job_id,
            "next_check_time": next_check_time.isoformat(),
            "next_task_id": next_task.id,
            "monitoring_metrics": {
                "count": metrics.monitoring_count,
                "duration": metrics.total_duration,
                "avg_interval": metrics.average_interval,
                "error_count": metrics.error_count,
                "retry_count": metrics.retry_count,
            },
            "task_metadata": task_metadata,
        }

    except TaskError:
        raise
    except Exception as e:
        logger.error(f"Failed to schedule next monitoring: {str(e)}")
        raise TaskError(
            f"Failed to schedule next monitoring: {str(e)}",
            error_type=TaskErrorType.TEMPORARY,
            original_error=e,
        )


# Celery beat schedule (if needed)
celery_app.conf.beat_schedule = {}
