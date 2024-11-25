from typing import List, Dict, Tuple, Any
import anthropic
import json
import os
from datetime import datetime, timedelta
import logging
from enum import Enum
from pathlib import Path
import pytz
import time

from anthropic.types.beta.messages.batch_create_params import Request

from src.python.anthropic_batch_request_util.config.settings import Settings, settings

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Anthropic batch processing status"""
    PROCESSING = "in_progress"
    ENDED = "ended"
    CANCELING = "canceling"
    UNKNOWN = "unknown"

    @classmethod
    def from_str(cls, value: str) -> 'BatchStatus':
        try:
            return next(status for status in cls if status.value == value)
        except StopIteration:
            return cls.UNKNOWN

class BatchResultType(Enum):
    """Anthropic batch result types"""
    SUCCEEDED = "succeeded"
    ERROR = "error"
    EXPIRED = "expired"
    CANCELED = "canceled"

class AnthropicErrorType(Enum):
    """Anthropic API specific error types"""
    RATE_LIMIT = "rate_limit"
    INVALID_REQUEST = "invalid_request"
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    SERVER_ERROR = "server_error"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"

    @classmethod
    def from_error(cls, error: Exception) -> 'AnthropicErrorType':
        error_str = str(error).lower()
        if "rate limit" in error_str:
            return cls.RATE_LIMIT
        elif "invalid request" in error_str:
            return cls.INVALID_REQUEST
        elif "authentication" in error_str:
            return cls.AUTHENTICATION
        elif "permission" in error_str:
            return cls.PERMISSION
        elif "not found" in error_str:
            return cls.NOT_FOUND
        elif "server error" in error_str:
            return cls.SERVER_ERROR
        elif "service unavailable" in error_str:
            return cls.SERVICE_UNAVAILABLE
        return cls.UNKNOWN

class BatchProcessingError(Exception):
    """Base exception for batch processing errors"""
    def __init__(
        self,
        message: str,
        error_type: AnthropicErrorType = AnthropicErrorType.UNKNOWN,
        original_error: Exception | None = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.original_error = original_error

class ChunkingError(BatchProcessingError):
    """Error during message chunking"""
    pass

class ValidationError(BatchProcessingError):
    """Error during request validation"""
    pass

class CacheControlError(BatchProcessingError):
    """Error during cache control preparation"""
    pass

class CacheType(Enum):
    """Cache types supported by the Anthropic batch API"""
    EPHEMERAL = "ephemeral"

    @classmethod
    def from_str(cls, value: str) -> 'CacheType':
        """Convert string to CacheType enum value"""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid cache type: {value}. Must be one of: {[e.value for e in cls]}")

class AnthropicBatchHandlerUtil:
    """
    Utility class for internal batch processing operations.
    Handles validation, caching, and data persistence operations.
    """
    
    @staticmethod
    def build_request_payload(
        system_prompt: str,
        messages: List[Dict[str, Any]],
        custom_id: str,
        model_string: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: list | None = None
    ) -> Dict[str, Any]:
        """
        Build single request payload for the Anthropic batch API.

        Args:
            system_prompt: System prompt for the request
            messages: List of messages for this request
            custom_id: Unique identifier for this request
            model_string: Full model identifier string
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            stop_sequences: Optional stop sequences

        Returns:
            Formatted request dictionary ready for the batch API

        Raises:
            ValidationError: If required parameters are invalid
        """
        try:
            # Validate input parameters
            if not isinstance(system_prompt, str) or not system_prompt:
                raise ValidationError("Invalid system prompt")
            if not isinstance(messages, list):
                raise ValidationError("Messages must be a list")
            if not isinstance(custom_id, str) or not custom_id:
                raise ValidationError("Invalid custom_id")
            if not isinstance(model_string, str) or not model_string:
                raise ValidationError("Invalid model_string")
            if not isinstance(max_tokens, int) or max_tokens <= 0:
                raise ValidationError("Invalid max_tokens")
            if not isinstance(temperature, float) or not 0 <= temperature <= 1:
                raise ValidationError("Temperature must be between 0 and 1")

            request: Dict[str, Any] = {
                "custom_id": custom_id,
                "params": {
                    "model": model_string,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                    "system": system_prompt
                }
            }

            if stop_sequences is not None:
                if not isinstance(stop_sequences, list):
                    raise ValidationError("stop_sequences must be a list")
                request["params"]["stop_sequences"] = stop_sequences

            logger.debug(
                f"Built request payload for {custom_id} with model {model_string}"
            )
            return request

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error building request payload: {str(e)}")
            raise ValidationError(f"Failed to build request payload: {str(e)}")

    @staticmethod
    def validate_batch_request(
        requests: List[Dict[str, Any]], 
        max_batch_size: int,
        max_payload_size: int
    ) -> None:
        """
        Validate batch request against configured limits.
        
        Args:
            requests: List of request dictionaries to validate
            max_batch_size: Maximum allowed batch size from settings
            max_payload_size: Maximum allowed payload size in bytes from settings

        Raises:
            ValidationError: If validation fails
        """
        try:
            # Check batch size limit
            if len(requests) > max_batch_size:
                raise ValidationError(
                    f"Batch size ({len(requests)}) exceeds maximum limit of {max_batch_size} requests"
                )

            # Validate payload size and structure
            try:
                payload = {"requests": requests}
                payload_size = len(json.dumps(payload).encode('utf-8'))
                
                if payload_size > max_payload_size:
                    raise ValidationError(
                        f"Request payload size ({payload_size / 1024 / 1024:.2f}MB) "
                        f"exceeds maximum limit of {max_payload_size / 1024 / 1024:.2f}MB"
                    )

                # Validate structure of each request
                for i, request in enumerate(requests):
                    if not isinstance(request, dict):
                        raise ValidationError(f"Request at index {i} is not a dictionary")
                    
                    required_fields = {"custom_id", "params"}
                    missing_fields = required_fields - set(request.keys())
                    if missing_fields:
                        raise ValidationError(
                            f"Request at index {i} is missing required fields: {missing_fields}"
                        )
                    
                    required_params = {"model", "messages", "system"}
                    missing_params = required_params - set(request["params"].keys())
                    if missing_params:
                        raise ValidationError(
                            f"Request at index {i} is missing required params: {missing_params}"
                        )

                logger.debug(
                    f"Batch request validated successfully: "
                    f"{len(requests)} requests, {payload_size / 1024 / 1024:.2f}MB"
                )

            except json.JSONDecodeError as e:
                raise ValidationError(f"Failed to serialize request payload: {str(e)}")

        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch validation: {str(e)}")
            raise ValidationError(f"Batch validation failed: {str(e)}")

    @staticmethod
    def prepare_cache_control(requests: List[Dict[str, Any]], cache_type: str) -> None:
        """
        Prepare system prompts for caching with configured cache type.
        Modifies the requests in place to add cache control settings.

        Args:
            requests: List of request dictionaries to prepare cache control for
            cache_type: Type of cache to use (from settings)

        Raises:
            CacheControlError: If cache type is invalid or system prompt structure is invalid
        """
        try:
            try:
                cache_type_enum = CacheType.from_str(cache_type)
            except ValueError as e:
                raise CacheControlError(f"Invalid cache type configuration: {str(e)}")

            for i, request in enumerate(requests):
                try:
                    params = request.get("params", {})
                    system = params.get("system")

                    if system is None:
                        logger.warning(f"Request {i} has no system prompt, skipping cache control")
                        continue

                    # Convert string system prompts to structured format
                    if isinstance(system, str):
                        structured_system = [{
                            "type": "text",
                            "text": system,
                            "cache_control": {"type": cache_type_enum.value}
                        }]
                        params["system"] = structured_system
                        logger.debug(f"Converted string system prompt to structured format for request {i}")

                    # Add cache control to list system prompts
                    elif isinstance(system, list):
                        for msg in system:
                            if not isinstance(msg, dict):
                                raise CacheControlError(
                                    f"Invalid system prompt structure in request {i}"
                                )
                            if "cache_control" not in msg:
                                msg["cache_control"] = {"type": cache_type_enum.value}
                        logger.debug(f"Added cache control to list system prompts for request {i}")

                    else:
                        raise CacheControlError(
                            f"Invalid system prompt type in request {i}: {type(system)}"
                        )

                except Exception as e:
                    raise CacheControlError(
                        f"Failed to prepare cache control for request {i}: {str(e)}"
                    )

            logger.debug(f"Cache control preparation completed for {len(requests)} requests")

        except CacheControlError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during cache control preparation: {str(e)}")
            raise CacheControlError(f"Cache control preparation failed: {str(e)}")

    @staticmethod
    def save_batch_request(
        job_id: str,
        requests: List[Dict[str, Any]],
        base_dir: str,
        request_file_prefix: str
    ) -> str:
        """
        Save batch request information to configured directory.
        Creates directory if it doesn't exist.

        Args:
            job_id: Batch job ID
            requests: List of request dictionaries
            base_dir: Base directory for saving request information
            request_file_prefix: Prefix for request files

        Returns:
            Path to saved request file

        Raises:
            BatchProcessingError: If saving fails
            ValidationError: If input parameters are invalid
        """
        try:
            # Validate input parameters
            if not job_id or not isinstance(job_id, str):
                raise ValidationError("Invalid job_id")
            if not requests or not isinstance(requests, list):
                raise ValidationError("Invalid requests data")
            if not base_dir or not isinstance(base_dir, str):
                raise ValidationError("Invalid base_dir")
            if not request_file_prefix or not isinstance(request_file_prefix, str):
                raise ValidationError("Invalid request_file_prefix")

            # Create directory if it doesn't exist
            save_dir = Path(base_dir)
            try:
                save_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise BatchProcessingError(f"Failed to create directory: {str(e)}")

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{request_file_prefix}_{timestamp}_{job_id}.json"
            save_path = save_dir / filename

            # Prepare batch information
            batch_info = {
                "job_id": job_id,
                "timestamp": datetime.now().isoformat(),
                "request_count": len(requests),
                "requests": requests
            }

            # Save to file with proper encoding and formatting
            try:
                with save_path.open('w', encoding='utf-8') as f:
                    json.dump(batch_info, f, indent=2, ensure_ascii=False)
            except IOError as e:
                raise BatchProcessingError(f"Failed to write file: {str(e)}")

            logger.debug(f"Batch request saved to: {save_path}")
            return str(save_path)

        except (ValidationError, BatchProcessingError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving batch request: {str(e)}")
            raise BatchProcessingError(f"Failed to save batch request: {str(e)}")

    @staticmethod
    def chunk_messages(
        messages_list: List[List[Dict[str, Any]]], 
        chunk_size: int
    ) -> List[List[List[Dict[str, Any]]]]:
        """
        Split large message lists into smaller chunks for batch processing.
        
        Args:
            messages_list: List of message lists to split
            chunk_size: Maximum size for each chunk

        Returns:
            List of chunked message lists, each chunk <= chunk_size

        Raises:
            ChunkingError: If chunking fails or validation fails
        """
        try:
            # Validate input parameters
            if not messages_list:
                raise ValidationError("Empty messages list")
            if chunk_size <= 0:
                raise ValidationError("Chunk size must be positive")
            if not isinstance(chunk_size, int):
                raise ValidationError("Chunk size must be an integer")

            # Calculate total size and number of chunks needed
            total_size = len(messages_list)
            num_chunks = (total_size + chunk_size - 1) // chunk_size

            logger.debug(
                f"Chunking {total_size} messages into {num_chunks} chunks "
                f"of size {chunk_size}"
            )

            # Split messages into chunks
            chunks = [
                messages_list[i:i + chunk_size]
                for i in range(0, total_size, chunk_size)
            ]

            # Validate resulting chunks
            for i, chunk in enumerate(chunks):
                if not chunk:
                    logger.warning(f"Empty chunk generated at index {i}")
                elif len(chunk) > chunk_size:
                    raise ChunkingError(
                        f"Chunk size validation failed: chunk {i} "
                        f"has size {len(chunk)} > {chunk_size}"
                    )

            logger.debug(
                f"Successfully split messages into {len(chunks)} chunks: "
                f"sizes={[len(chunk) for chunk in chunks]}"
            )

            return chunks

        except (ValidationError, ChunkingError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during message chunking: {str(e)}")
            raise ChunkingError(f"Message chunking failed: {str(e)}")

    @staticmethod
    def process_large_batch(
        handler: 'AnthropicBatchHandler',
        system_prompt: str,
        messages_list: List[List[Dict[str, Any]]],
        chunk_size: int | None = None,
        retry_strategy: Dict[str, Any] | None = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Process large batch by splitting into chunks and executing each chunk.
        
        Args:
            handler: AnthropicBatchHandler instance for executing requests
            system_prompt: System prompt for all requests
            messages_list: List of message lists to process
            chunk_size: Optional chunk size override (uses settings if not provided)
            retry_strategy: Optional retry configuration for failed chunks
                {
                    'max_retries': int,
                    'retry_delay': int,
                    'backoff_factor': float
                }
            **kwargs: Additional arguments for execute_batch_with_monitoring

        Returns:
            List of execution results for each chunk

        Raises:
            BatchProcessingError: If batch processing fails
            ValidationError: If input parameters are invalid
        """
        try:
            retry_config = retry_strategy or {
                'max_retries': 3,
                'retry_delay': 60,
                'backoff_factor': 2.0
            }
            # Validate input parameters
            if not isinstance(system_prompt, str) or not system_prompt:
                raise ValidationError("Invalid system prompt")
            if not messages_list:
                raise ValidationError("Empty messages list")

            # Use settings chunk size if not provided
            effective_chunk_size = chunk_size or settings.batch.chunk_size
            logger.debug(
                f"Starting large batch processing with chunk size {effective_chunk_size}"
            )

            # Split messages into chunks
            chunks = AnthropicBatchHandlerUtil.chunk_messages(
                messages_list=messages_list,
                chunk_size=effective_chunk_size
            )

            results = []
            total_chunks = len(chunks)

            # Process each chunk
            for i, chunk in enumerate(chunks, 1):
                retries = 0
                while retries <= retry_config['max_retries']:
                    try:
                        logger.debug(f"Processing chunk {i}/{total_chunks} (size: {len(chunk)})")

                        # Generate unique prefix for this chunk
                        chunk_prefix = f"{kwargs.get('custom_id_prefix', 'request')}_{i}"
                        chunk_kwargs = {**kwargs, 'custom_id_prefix': chunk_prefix}

                        # Execute batch for this chunk
                        result = handler.execute_batch_with_monitoring(
                            system_prompt=system_prompt,
                            messages_list=chunk,
                            **chunk_kwargs
                        )

                        results.append({
                            'chunk_index': i,
                            'chunk_size': len(chunk),
                            'result': result
                        })

                        logger.debug(f"Chunk {i}/{total_chunks} processed successfully")
                        break
                    except Exception as e:
                        retries += 1
                        if retries > retry_config['max_retries']:
                            raise
                        delay = retry_config['retry_delay'] * (retry_config['backoff_factor'] ** (retries - 1))
                        logger.warning(f"Chunk {i} failed, retrying in {delay}s: {str(e)}")
                        time.sleep(delay)

            # Summarize results
            successful_chunks = sum(1 for r in results if 'error' not in r)
            logger.info(
                f"Large batch processing completed: "
                f"{successful_chunks}/{total_chunks} chunks successful"
            )

            return results

        except (ValidationError, ChunkingError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during large batch processing: {str(e)}")
            raise BatchProcessingError(f"Large batch processing failed: {str(e)}")

class AnthropicBatchHandler:
    """
    Main handler for Anthropic batch API operations.
    Provides public interface for batch request execution and monitoring.
    """
    
    def __init__(self, external_settings: Settings | None = None):
        """
        Initialize batch handler.
        
        Args:
            settings: Optional custom settings
        """
        self.settings: Settings = settings or Settings.load()
        self.util = AnthropicBatchHandlerUtil()
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")

        self.client = anthropic.Anthropic(
            api_key=api_key,
            default_headers={
                "anthropic-version": settings.api.version,
                "anthropic-beta": ",".join(settings.api.beta_features)
            }
        )

    def build_requests(
        self,
        system_prompt: str,
        messages_list: List[List[dict]],
        custom_id_prefix: str = "request",
        enable_prompt_cache: bool | None = None
    ) -> List[Dict[str, Any]]:
        """
        Build batch requests from message lists.

        Args:
            system_prompt: System prompt for all requests
            messages_list: List of message lists for each request
            custom_id_prefix: Prefix for request IDs
            enable_prompt_cache: Override default prompt cache setting from config.
                               If None, uses setting from config.

        Returns:
            List of formatted request dictionaries ready for batch API

        Raises:
            ValidationError: If input parameters are invalid
        """
        try:
            # Validate input parameters
            if not isinstance(system_prompt, str) or not system_prompt:
                raise ValidationError("Invalid system prompt")
            if not isinstance(messages_list, list) or not messages_list:
                raise ValidationError("Invalid messages list")
            if not isinstance(custom_id_prefix, str) or not custom_id_prefix:
                raise ValidationError("Invalid custom_id_prefix")

            requests = []

            for i, messages in enumerate(messages_list):
                try:
                    # Generate unique custom ID for each request
                    custom_id = f"{custom_id_prefix}_{i}"

                    # Build request using utility method
                    request = self.util.build_request_payload(
                        system_prompt=system_prompt,
                        messages=messages,
                        custom_id=custom_id,
                        model_string=self.settings.model.name,
                        max_tokens=self.settings.model.max_tokens,
                        temperature=self.settings.model.temperature,
                        stop_sequences=self.settings.model.stop_sequences
                    )
                    requests.append(request)

                    logger.debug(
                        f"Built request {i+1}/{len(messages_list)} with ID: {custom_id}"
                    )

                except ValidationError as e:
                    logger.error(f"Failed to build request {i}: {str(e)}")
                    raise ValidationError(f"Failed to build request {i}: {str(e)}")

            # Handle prompt caching if enabled
            use_cache = (
                enable_prompt_cache 
                if enable_prompt_cache is not None 
                else self.settings.batch.enable_prompt_cache
            )
            
            if use_cache:
                try:
                    self.util.prepare_cache_control(
                        requests=requests,
                        cache_type=self.settings.batch.cache_type
                    )
                    logger.debug("Prompt caching prepared for requests")
                except CacheControlError as e:
                    logger.error(f"Failed to prepare cache control: {str(e)}")
                    raise

            logger.info(
                f"Successfully built {len(requests)} requests with prefix '{custom_id_prefix}'"
                f" (prompt_cache: {use_cache})"
            )
            return requests

        except (ValidationError, CacheControlError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error building requests: {str(e)}")
            raise ValidationError(f"Failed to build requests: {str(e)}")

    def execute_batch(
        self, 
        system_prompt: str,
        messages_list: List[List[dict]],
        enable_prompt_cache: bool | None = None,
        custom_id_prefix: str = "request"
    ) -> str:
        """
        Execute a batch request.

        Args:
            system_prompt: System prompt for all requests
            messages_list: List of message lists for each request
            enable_prompt_cache: Override default cache setting from config
            custom_id_prefix: Prefix for request IDs

        Returns:
            Batch job ID for status checking

        Raises:
            ValidationError: If batch size or payload exceeds limits
            BatchProcessingError: If batch execution fails
        """
        try:
            # Build requests with cache configuration
            requests = self.build_requests(
                system_prompt=system_prompt,
                messages_list=messages_list,
                custom_id_prefix=custom_id_prefix,
                enable_prompt_cache=enable_prompt_cache
            )

            # Validate batch request against configured limits
            self.util.validate_batch_request(
                requests=requests,
                max_batch_size=self.settings.batch.max_size,
                max_payload_size=self.settings.api.max_payload_size
            )

            try:
                # Execute batch request
                message_batch = self.client.beta.messages.batches.create(
                    betas = self.settings.api.beta_features,
                    requests=[Request(custom_id=r["custom_id"], params=r["params"]) for r in requests]
                )
                
                logger.info(f"Batch request submitted successfully: {message_batch.id}")

                # Save request information for logging/debugging
                save_path = self.util.save_batch_request(
                    job_id=message_batch.id,
                    requests=requests,
                    base_dir=str(self.settings.storage.base_dir / self.settings.storage.subdirs["requests"]),
                    request_file_prefix=self.settings.storage.request_file_prefix
                )
                
                logger.debug(f"Batch request saved to: {save_path}")
                
                return message_batch.id

            except Exception as e:
                logger.error(f"Failed to execute batch request: {str(e)}")
                raise BatchProcessingError(f"Batch execution failed: {str(e)}")

        except (ValidationError, BatchProcessingError):
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch execution: {str(e)}")
            raise BatchProcessingError(f"Batch execution failed: {str(e)}")

    def check_batch_status(self, job_id: str) -> Tuple[str, Any, Dict[str, Dict[str, int]]]:
        """
        Check the status of a batch request.
        
        Args:
            job_id: The ID of the batch request to check
            
        Returns:
            Tuple containing:
            - status (str): The status of the batch
            - results (Any): The results if available
            - metrics (Dict[str, Dict[str, int]]): Metrics about the batch processing
        """
        try:
            message_batch = self.client.beta.messages.batches.retrieve(job_id)
            
            # Get results from results_url if available
            results = None
            if message_batch.processing_status == "ended" and message_batch.results_url:
                # Fetch results from results_url
                results = self._fetch_results_from_url(message_batch.results_url)
            
            # Convert metrics to expected format
            metrics = {
                "request_counts": {
                    "processing": message_batch.request_counts.processing,
                    "succeeded": message_batch.request_counts.succeeded,
                    "errored": message_batch.request_counts.errored,
                    "canceled": message_batch.request_counts.canceled,
                    "expired": message_batch.request_counts.expired
                }
            }
            
            return message_batch.processing_status, results, metrics
            
        except Exception as e:
            logger.error(f"Failed to check batch status: {str(e)}")
            raise BatchProcessingError(
                f"Failed to check batch status: {str(e)}",
                error_type=AnthropicErrorType.from_error(e),
                original_error=e
            )

    def _fetch_results_from_url(self, results_url: str) -> List[Any]:
        """
        Fetch results from the provided results URL using the Anthropic client.
        
        Args:
            results_url: URL to fetch the results from
            
        Returns:
            List of batch results
            
        Raises:
            BatchProcessingError: If fetching results fails
        """
        try:
            # Get batch ID from results_url
            # Expected format: ".../messages/batches/{batch_id}/results"
            batch_id = results_url.split('/')[-2]
            
            # Use the client's results method to fetch results
            results = []
            for result in self.client.beta.messages.batches.results(batch_id):
                results.append(result)
                
            logger.debug(f"Successfully fetched {len(results)} results from batch {batch_id}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to fetch results from URL: {str(e)}")
            raise BatchProcessingError(
                f"Failed to fetch results: {str(e)}",
                error_type=AnthropicErrorType.from_error(e),
                original_error=e
            )

    def execute_batch_with_monitoring(
        self,
        system_prompt: str,
        messages_list: List[List[dict]],
        enable_prompt_cache: bool | None = None,
        custom_id_prefix: str = "request"
    ) -> Dict:
        """
        Execute batch request with monitoring.

        Args:
            system_prompt: System prompt for all requests
            messages_list: List of message lists for each request
            enable_prompt_cache: Override default cache setting from config
            custom_id_prefix: Prefix for request IDs

        Returns:
            Dict containing:
                - status: Submission status
                - job_id: Batch job ID
                - result_path: Path where results will be saved
                - start_time: Batch start time
                - scheduled_time: Next monitoring time
                - task_id: Celery task ID
                - timezone: Timezone used for scheduling

        Raises:
            BatchProcessingError: If batch execution or monitoring setup fails
        """
        try:
            # Execute batch request
            job_id = self.execute_batch(
                system_prompt=system_prompt,
                messages_list=messages_list,
                enable_prompt_cache=enable_prompt_cache,
                custom_id_prefix=custom_id_prefix
            )
            
            # Set up monitoring task parameters
            timezone = pytz.timezone(self.settings.worker.worker.timezone)
            start_time = datetime.now(timezone)
            scheduled_time = start_time + timedelta(seconds=self.settings.worker.monitor.initial_delay)
            
            # Generate result file path
            timestamp = start_time.strftime("%Y%m%d_%H%M%S")
            result_path = str(
                self.settings.storage.base_dir / 
                self.settings.storage.subdirs["results"] /
                f"{self.settings.storage.result_file_prefix}_{timestamp}_{job_id}.json"
            )
            
            logger.info(f"Scheduling monitoring task for job_id: {job_id}")
            logger.info(f"Start time: {start_time.isoformat()}")
            logger.info(f"Scheduled time: {scheduled_time.isoformat()}")
            
            try:
                # Schedule monitoring task
                from .batch_tasks import monitor_batch
                task = monitor_batch.apply_async(
                    args=[job_id, result_path],
                    kwargs={
                        "start_time": start_time.isoformat(),
                    },
                    eta=scheduled_time
                )
                
                logger.info(f"Monitoring task scheduled with task_id: {task.id}")
                
                return {
                    "status": "submitted",
                    "job_id": job_id,
                    "result_path": result_path,
                    "start_time": start_time.isoformat(),
                    "scheduled_time": scheduled_time.isoformat(),
                    "task_id": task.id,
                    "timezone": str(timezone)
                }
                
            except Exception as e:
                logger.error(f"Failed to schedule monitoring task: {str(e)}")
                raise BatchProcessingError(f"Failed to schedule monitoring: {str(e)}")
                
        except Exception as e:
            logger.error(f"Batch submission failed: {str(e)}", exc_info=True)
            return {
                "status": "error",
                "error": str(e)
            }

    def get_batch_metrics(self, job_id: str) -> Dict[str, Any]:
        """
        Get detailed metrics for a batch job.
        """
        try:
            message_batch = self.client.beta.messages.batches.retrieve(job_id)
            
            # Calculate total requests and success rate
            total_requests = sum([
                message_batch.request_counts.processing,
                message_batch.request_counts.succeeded,
                message_batch.request_counts.errored,
                message_batch.request_counts.canceled,
                message_batch.request_counts.expired
            ])
            
            completed_requests = sum([
                message_batch.request_counts.succeeded,
                message_batch.request_counts.errored,
                message_batch.request_counts.expired
            ])
            
            success_rate = (
                (message_batch.request_counts.succeeded / completed_requests * 100)
                if completed_requests > 0 else 0
            )
            
            return {
                "request_counts": {
                    "processing": message_batch.request_counts.processing,
                    "succeeded": message_batch.request_counts.succeeded,
                    "errored": message_batch.request_counts.errored,
                    "canceled": message_batch.request_counts.canceled,
                    "expired": message_batch.request_counts.expired,
                    "total": total_requests
                },
                "success_rate": success_rate,
                "status": message_batch.processing_status,
                "has_errors": message_batch.request_counts.errored > 0,
                "error_rate": (
                    message_batch.request_counts.errored / completed_requests * 100
                    if completed_requests > 0 else 0
                )
            }
            
        except Exception as e:
            logger.error(f"Failed to get batch metrics for {job_id}: {str(e)}")
            raise BatchProcessingError(f"Failed to get batch metrics: {str(e)}")

    def serialize_batch_results(self, results: List[Any]) -> List[Dict[str, Any]]:
        """
        Serialize batch results into a consistent format.
        """
        serialized_results = []
        
        try:
            for result in results:
                serialized_result = {
                    'custom_id': result.custom_id,
                    'result': {
                        'type': result.result.type,
                    }
                }
                
                # Handle successful results
                if result.result.type == 'succeeded':
                    message = result.result.message
                    serialized_result['result'].update({
                        'message': {
                            'id': message.id,
                            'type': message.type,
                            'role': message.role,
                            'content': [
                                {
                                    'type': content_block.type,
                                    'text': content_block.text
                                }
                                for content_block in message.content
                            ],
                            'model': message.model,
                            'stop_reason': message.stop_reason,
                            'stop_sequence': message.stop_sequence,
                            'usage': {
                                'input_tokens': message.usage.input_tokens,
                                'output_tokens': message.usage.output_tokens
                            }
                        }
                    })
                # Handle error results
                elif result.result.type == 'error':
                    serialized_result['result'].update({
                        'error': {
                            'type': result.result.error.type,
                            'message': result.result.error.message
                        }
                    })
                
                serialized_results.append(serialized_result)
                
            return serialized_results
            
        except Exception as e:
            logger.error(f"Failed to serialize batch results: {str(e)}")
            raise BatchProcessingError(f"Failed to serialize results: {str(e)}") 