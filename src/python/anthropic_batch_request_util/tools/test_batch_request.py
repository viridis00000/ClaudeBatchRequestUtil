import sys
import logging
from pathlib import Path
from typing import List, Dict

# Add parent directory to path to import batch handler
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from ..batch_handler import AnthropicBatchHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_test_messages() -> List[List[Dict]]:
    """
    Create a minimal test message list for batch processing.
    """
    return [
        [  # First conversation
            {"role": "user", "content": "What is 2+2?"}
        ],
        [  # Second conversation
            {"role": "user", "content": "What is the capital of France?"}
        ],
    ]


def main():
    """
    Execute a minimal test batch request.
    """
    try:
        # Initialize batch handler
        handler = AnthropicBatchHandler()

        # Create test messages
        messages_list = create_test_messages()
        system_prompt = (
            "You are a helpful AI assistant. Provide clear and concise answers."
        )

        logger.info("Executing batch request with test messages...")

        # Execute batch with monitoring
        result = handler.execute_batch_with_monitoring(
            system_prompt=system_prompt,
            messages_list=messages_list,
            custom_id_prefix="test_batch",
        )

        logger.info("Batch request submitted successfully:")
        logger.info(f"Job ID: {result['job_id']}")
        logger.info(f"Result will be saved to: {result['result_path']}")
        logger.info(f"Monitoring task ID: {result['task_id']}")

    except Exception as e:
        logger.error(f"Error executing test batch request: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
