import os, re, logging


class MultipleHeaderFilter(logging.Filter):
    def __init__(self, patterns_to_filter):
        super().__init__()
        self.patterns_to_filter = [re.compile(pattern) for pattern in patterns_to_filter]

    def filter(self, record):
        message = record.getMessage()
        return not any(pattern.search(message) for pattern in self.patterns_to_filter)


def setup_logging(save_path, patterns_to_filter=None):
    """
    Configure logging to file and console, ignoring specified message patterns.

    :param save_path: Path where the log file will be saved
    :param patterns_to_filter: List of regex patterns for messages to be filtered out
    """

    if patterns_to_filter is None:
        patterns_to_filter = [
            r"^HTTP Request:",
            r"^OpenAI API response:",
            r"^Retrying request",
            # Add more patterns here as needed
        ]

    log_path = f"{save_path}/output.log"
    if os.path.exists(log_path):
        os.remove(log_path)

    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    # Create formatter and add it to the handlers
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add filter to console handler not file handler
    multiple_header_filter = MultipleHeaderFilter(patterns_to_filter)
    # file_handler.addFilter(multiple_header_filter)
    console_handler.addFilter(multiple_header_filter)

    # Get the root logger and set its level
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove any existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add the new handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
