import logging

LOGGING_LEVEL = logging.DEBUG


def setup_logger(logger_name, level=LOGGING_LEVEL):
    """
    A basic logger setup that logs to console.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a formatter
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(console_handler)

    return logger
