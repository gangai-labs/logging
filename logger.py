import json
import logging
import logging.handlers
import os
import re

from datetime import datetime
from pathlib import Path
from colorlog import ColoredFormatter
from pydantic import Field, field_validator, ValidationError
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any


class LoggerConfig(BaseSettings):
    """Logger configuration using Pydantic with validation"""

    # Logging settings
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    LOG_FILE: Optional[str] = Field(
        default=None,
        description="Log file path (optional). Relative paths are resolved from project root."
    )
    ENABLE_SEVERITY_FILES: bool = Field(
        default=True,
        description="Enable separate log files for WARNING, ERROR, and CRITICAL messages"
    )
    SEVERITY_FILES_DIR: str = Field(
        default="logs/severity",
        description="Directory for severity-specific log files"
    )
    MAX_LOG_SIZE_MB: int = Field(
        default=10,
        description="Maximum log file size in MB before rotation"
    )
    BACKUP_COUNT: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )

    class Config:
        env_file = ".env"
        case_sensitive = False

    @field_validator('LOG_LEVEL')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level - SILENT"""
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        v_upper = v.upper()

        if v_upper not in valid_levels:
            raise ValueError(f'Invalid LOG_LEVEL: {v}. Must be one of {valid_levels}')

        return v_upper

    @field_validator('LOG_FILE')
    @classmethod
    def validate_log_file(cls, v: Optional[str]) -> Optional[str]:
        """Validate log file path - SILENT"""
        if v is None or v.strip() == "":
            return None

        v = v.strip()

        # If path is not absolute, resolve it relative to project root
        if not os.path.isabs(v):
            # Get project root (directory where main.py is located)
            project_root = Path(__file__).parent.parent.absolute()
            v = str(project_root / v)

        if not Path(v).suffix:
            v = f"{v}.log"

        log_dir = os.path.dirname(v)

        try:
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            if log_dir:
                test_file = os.path.join(log_dir, '.write_test')
                try:
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                except (PermissionError, OSError) as e:
                    raise ValueError(f'Cannot write to log directory {log_dir}: {e}')

        except Exception as e:
            raise ValueError(f'Invalid LOG_FILE path: {v}. Error: {e}')

        return v

    @field_validator('SEVERITY_FILES_DIR')
    @classmethod
    def validate_severity_dir(cls, v: str) -> str:
        """Validate severity files directory"""
        v = v.strip()

        # If path is not absolute, resolve it relative to project root
        if not os.path.isabs(v):
            project_root = Path(__file__).parent.parent.absolute()
            v = str(project_root / v)

        return v

    @field_validator('MAX_LOG_SIZE_MB')
    @classmethod
    def validate_max_log_size(cls, v: int) -> int:
        """Validate max log size"""
        if v <= 0:
            raise ValueError('MAX_LOG_SIZE_MB must be positive')
        if v > 1000:  # 1GB limit
            raise ValueError('MAX_LOG_SIZE_MB cannot exceed 1000 MB')
        return v

    @field_validator('BACKUP_COUNT')
    @classmethod
    def validate_backup_count(cls, v: int) -> int:
        """Validate backup count"""
        if v < 0:
            raise ValueError('BACKUP_COUNT must be non-negative')
        if v > 50:  # Reasonable limit
            raise ValueError('BACKUP_COUNT cannot exceed 50')
        return v


class SeverityFilter(logging.Filter):
    """Filter to only allow specific log levels"""

    def __init__(self, level):
        super().__init__()
        self.level = level

    def filter(self, record):
        return record.levelno == self.level


class JsonFormatter(logging.Formatter):
    """Custom formatter that outputs JSON for ERROR and CRITICAL levels"""

    def format(self, record):
        # Only format as JSON for ERROR and CRITICAL levels
        if record.levelno == logging.ERROR:
            try:
                exc_info = record.exc_info

                if exc_info:

                    exc_type, exc_value, exc_tb = exc_info

                    # Get traceback information
                    if exc_tb:

                        frame = exc_tb.tb_frame
                        filename = frame.f_code.co_filename
                        lineno = exc_tb.tb_lineno
                        function = frame.f_code.co_name


                    else:
                        filename = lineno = function = "unknown"

                    # Build JSON context
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    error_context = {
                        "error_type": exc_type.__name__ if exc_type else "Unknown",
                        "error_message": str(exc_value),
                        "file": filename,
                        "line": lineno,
                        "function": function,
                        "traceback": self.formatException(exc_info),
                        "timestamp": timestamp,
                        "level": record.levelname,
                        "logger": record.name,
                        "message": record.getMessage()
                    }

                    json_output = json.dumps(error_context, indent=2)
                    return f"{timestamp} - [{record.name}] {record.levelname}:\n{json_output}"
                else:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                    return  f"{timestamp} - [{record.name}] {record.levelname} -  {record.getMessage()}"
            except Exception as e:
                # Fallback to normal formatting if JSON fails
                return super().format(record)

        # For non-error levels, use normal formatting
        return super().format(record)


class ColoredJsonFormatter(JsonFormatter):
    """Custom formatter that outputs colored JSON for console output"""

    COLORS = {
        'ERROR': '\033[91m',  # Red
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        formatted_message = super().format(record)

        # Add color for error levels in console
        if record.levelno == logging.ERROR:
            color = self.COLORS.get(record.levelname, self.COLORS['ERROR'])
            return f"{color}{formatted_message}{self.COLORS['RESET']}"

        return formatted_message


class Logger:
    def __init__(self):
        self._logger_instances = {}  # Global logger registry to prevent duplicates
        self._complete_log_handler = None  # Single complete log handler instance
        self._severity_handlers = {}  # Shared severity handlers

    def _setup_severity_handlers(self, config: LoggerConfig):
        """Setup shared severity-specific file handlers"""
        if not config.ENABLE_SEVERITY_FILES or self._severity_handlers:
            return

        severity_levels = {
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        try:
            # Ensure severity directory exists
            os.makedirs(config.SEVERITY_FILES_DIR, exist_ok=True)

            # Test write permissions
            test_file = os.path.join(config.SEVERITY_FILES_DIR, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)

            # Create JSON formatter for severity files
            json_formatter = JsonFormatter(
                '%(asctime)s - [%(name)s] %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            for level_name, level_value in severity_levels.items():
                log_file_path = os.path.join(config.SEVERITY_FILES_DIR, f"{level_name.lower()}.log")

                handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    mode='a',
                    maxBytes=config.MAX_LOG_SIZE_MB * 1024 * 1024,
                    backupCount=config.BACKUP_COUNT,
                    encoding='utf-8'
                )
                handler.setFormatter(json_formatter)
                handler.setLevel(level_value)
                handler.addFilter(SeverityFilter(level_value))

                self._severity_handlers[level_name] = handler

        except Exception as e:
            print(f"⚠️  Warning: Could not set up severity-specific log files: {e}")
            # Continue without severity files

    def setup_logger(self, config: Optional[LoggerConfig] = None, logger_name: str = "main_logger"):
        """Setup logger with validated Pydantic configuration - Singleton pattern"""
        # Return existing logger if already created
        if logger_name in self._logger_instances:
            return self._logger_instances[logger_name]

        # Load and validate config if not provided
        if config is None:
            try:
                config = LoggerConfig()
            except ValidationError as e:
                print("❌ Logger configuration validation failed fix .env :")
                for error in e.errors():
                    print(f"  - {error['loc'][0]}: {error['msg']}")
                # Fall back to basic config
                config = LoggerConfig(
                    LOG_LEVEL="INFO",
                    LOG_FILE=None,
                    ENABLE_SEVERITY_FILES=False
                )
                return None

        # Setup shared handlers
        self._setup_severity_handlers(config)

        # Main application logger
        logger = logging.getLogger(logger_name)

        # Clear existing handlers to prevent duplication
        logger.handlers.clear()
        logger.propagate = False

        # Set log level
        level_mapping = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        logger.setLevel(level_mapping[config.LOG_LEVEL])

        # Formatters
        normal_formatter = logging.Formatter(
            '%(asctime)s - [%(name)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        color_formatter = ColoredFormatter(
            '%(log_color)s%(asctime)s - [%(name)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',
            }
        )

        json_formatter = JsonFormatter()
        colored_json_formatter = ColoredJsonFormatter()

        # Console handler - use colored formatter for all levels
        console_handler = logging.StreamHandler()

        # For console, use colored JSON for errors and normal color for others
        class ConsoleLevelAwareFormatter(logging.Formatter):
            def format(self, record):
                if record.levelno == logging.ERROR:

                    return colored_json_formatter.format(record)
                else:

                    return color_formatter.format(record)

        console_handler.setFormatter(ConsoleLevelAwareFormatter())
        logger.addHandler(console_handler)

        # File handler for specific logger (if specified)
        if config.LOG_FILE:
            try:
                file_handler = logging.handlers.RotatingFileHandler(
                    config.LOG_FILE,
                    mode='a',
                    maxBytes=config.MAX_LOG_SIZE_MB * 1024 * 1024,
                    backupCount=config.BACKUP_COUNT,
                    encoding='utf-8'
                )

                # For files, use JSON for errors and normal format for others
                class FileLevelAwareFormatter(logging.Formatter):
                    def format(self, record):
                        if record.levelno >= logging.ERROR:
                            return json_formatter.format(record)
                        else:
                            return normal_formatter.format(record)

                file_handler.setFormatter(FileLevelAwareFormatter())
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"⚠️  Error: Could not set up file logging: {e}")
                logger.warning(f"File logging disabled due to error: {e}")

        # Add complete log handler (shared across all loggers)
        if self._complete_log_handler is None:
            try:
                file_path = Path(__file__).parent.parent.absolute()
                complete_log_path = f"{file_path}/logs/complete_log.log"

                # Ensure directory exists
                os.makedirs(os.path.dirname(complete_log_path), exist_ok=True)

                self._complete_log_handler = logging.handlers.RotatingFileHandler(
                    complete_log_path,
                    mode='a',
                    maxBytes=config.MAX_LOG_SIZE_MB * 1024 * 1024,
                    backupCount=config.BACKUP_COUNT,
                    encoding='utf-8'
                )

                # For complete log, use JSON for errors and normal format for others
                class CompleteLogLevelAwareFormatter(logging.Formatter):
                    def format(self, record):
                        if record.levelno >= logging.ERROR:
                            return json_formatter.format(record)
                        else:
                            return normal_formatter.format(record)

                self._complete_log_handler.setFormatter(CompleteLogLevelAwareFormatter())
            except Exception as e:
                print(f"⚠️  Error: Could not set up complete log: {e}")
                # Don't add handler if it fails

        # Add complete log handler if it was successfully created
        if self._complete_log_handler:
            logger.addHandler(self._complete_log_handler)

        # Add severity-specific handlers
        for handler in self._severity_handlers.values():
            logger.addHandler(handler)

        # Store in registry
        self._logger_instances[logger_name] = logger



        return logger

    def close_logger(self, logger_name: str = None):
        """Close and clean up logger handlers"""
        if logger_name:
            # Close specific logger
            if logger_name in self._logger_instances:
                logger = self._logger_instances[logger_name]
                for handler in logger.handlers[:]:  # Copy list to avoid modification during iteration
                    handler.close()
                    logger.removeHandler(handler)
                del self._logger_instances[logger_name]
        else:
            # Close all loggers
            for name, logger in list(self._logger_instances.items()):
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
            self._logger_instances.clear()

            # Close shared handlers
            if self._complete_log_handler:
                self._complete_log_handler.close()
                self._complete_log_handler = None

            for handler in self._severity_handlers.values():
                handler.close()
            self._severity_handlers.clear()

    def close_all_loggers(self):
        """Close all loggers and handlers - alias for close_logger()"""
        for instance in self.list_loggers():
            self.close_logger(instance)


    def list_loggers(self):
        """Return list of active logger names"""
        return list(self._logger_instances.keys())


if __name__ == '__main__':
    
    # Initialize the logger manager
    logger_manager = Logger()

    # Example 2: Create logger with custom configuration programmatically
    client_logger_config = LoggerConfig(
        LOG_LEVEL="DEBUG",
        LOG_FILE="logs/client.log",
        ENABLE_SEVERITY_FILES=True,
        SEVERITY_FILES_DIR="logs/severity"
    )
    client_logger = logger_manager.setup_logger(config=client_logger_config, logger_name="client")
    class CreateErr:
        def zerodiverror(self):
            try:
                result = 10 / 0  # This will cause ZeroDivisionError
            except Exception as e:
                client_logger.exception("Division by zero occurred in calculation")
    CreateErr().zerodiverror()
    # Example 2: Basic demo
    client_logger.error('my error')
    client_logger.critical('my critical')
    client_logger.info('my info')
    client_logger.debug('my debug')


    client_logger.info('123')
    # Example 3: Create another logger with different configuration
    db_logger_config = LoggerConfig(
        LOG_LEVEL="INFO",
        LOG_FILE="logs/database.log",
        ENABLE_SEVERITY_FILES=True
    )
    db_logger = logger_manager.setup_logger(config=db_logger_config, logger_name="database")
    if db_logger:
        db_logger.info("Database logger configured")
        db_logger.debug("This debug message won't appear due to INFO level")
        db_logger.critical("Database critical error - will be in both database.log and critical.log")

    # Example 4: Console-only logger without severity files
    console_only_config = LoggerConfig(
        LOG_LEVEL="DEBUG",
        LOG_FILE=None,
        ENABLE_SEVERITY_FILES=False
    )
    console_logger = logger_manager.setup_logger(config=console_only_config, logger_name="console")
    if console_logger:
        console_logger.debug("Console-only logger message")
        console_logger.error("This error won't be saved to severity files")

    # Example 5: Test all severity levels with rotation
    print("\n=== Testing all severity levels with log rotation ===")
    test_config = LoggerConfig(
        LOG_LEVEL="DEBUG",
        LOG_FILE="logs/test.log",
        ENABLE_SEVERITY_FILES=True,
        SEVERITY_FILES_DIR="logs/alerts",
        MAX_LOG_SIZE_MB=1,  # Small size for testing rotation
        BACKUP_COUNT=3
    )
    test_logger = logger_manager.setup_logger(config=test_config, logger_name="test")
    if test_logger:
        test_logger.debug("Debug message - only in test.log and complete_log.log")
        test_logger.info("Info message - only in test.log and complete_log.log")
        test_logger.warning("Warning message - in test.log, complete_log.log, AND alerts/warning.log")
        test_logger.error("Error message - in test.log, complete_log.log, AND alerts/error.log")
        test_logger.critical("Critical message - in test.log, complete_log.log, AND alerts/critical.log")

    # Example 6: Logger cleanup
    print("\n=== Logger cleanup ===")
    print(f"Active loggers before cleanup: {logger_manager.list_loggers()}")

    # Close specific logger
    logger_manager.close_logger("test")
    #print(f"Active loggers after closing 'test': {logger_manager.list_loggers()}")

    # Close all loggers
    logger_manager.close_all_loggers()


    print(f"Active loggers after closing all: {logger_manager.list_loggers()}")
    #
    print("\nCheck your log directories:")
    print("- logs/complete_log.log (all messages)")
    print("- logs/client.log, logs/database.log, logs/test.log (logger-specific)")
    print("- logs/severity/warning.log (all WARNING messages)")
    print("- logs/severity/error.log (all ERROR messages)")
    print("- logs/severity/critical.log (all CRITICAL messages)")
    print("- logs/alerts/warning.log, logs/alerts/error.log, logs/alerts/critical.log (test logger alerts)")
    print("- Look for .1, .2, .3 backup files if rotation occurred!")
