# Python Logging Wrapper

A simplified logging wrapper that automates file creation and provides structured logging with minimal setup.

## 🚀 Features

- ✅ **Automatic log file creation** and directory setup
- ✅ **Colored console output** for better readability  
- ✅ **JSON-formatted error logs** with traceback information
- ✅ **Log rotation** with configurable size limits
- ✅ **Environment-based configuration** using `.env` files
- ✅ **Multiple logger instances** with different configurations
- ✅ **100% compatible** with standard Python logging

## 📦 Installation

```bash
pip install colorlog pydantic-settings
```

## 🏃 Quick Start

### Basic Usage

```python
from your_logging_module import Logger

# Initialize logger (automatically reads from .env if available)
logger_manager = Logger()

# Get a logger instance
logger = logger_manager.setup_logger()

# Use like standard logging
logger.info("Application started")
logger.debug("Debug information")
logger.warning("This is a warning")
logger.error("An error occurred")

# Automatic JSON formatting for exceptions
try:
    result = 10 / 0
except Exception:
    logger.exception("Division failed")  # Auto-formatted as JSON
```

### Environment Configuration

Create a `.env` file in your project root:

```ini
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
ENABLE_SEVERITY_FILES=true
MAX_LOG_SIZE_MB=10
BACKUP_COUNT=5
```

## 📋 Best Practices

### Per-Module Logging

Create a dedicated logger for each module to track message sources:

```python
# In each Python file
from your_logging_module import Logger

logger_manager = Logger()
logger = logger_manager.setup_logger(logger_name=__name__)

def my_function():
    logger.info("Function executed")  # Shows module name in logs
```

### Class-Based Logging

For object-oriented code:

```python
class DatabaseService:
    def __init__(self):
        self.logger = Logger().setup_logger(
            logger_name=f"database.{self.__class__.__name__}"
        )
    
    def query(self, sql):
        self.logger.debug(f"Executing: {sql}")
        try:
            # Database operations
            self.logger.info("Query successful")
        except Exception:
            self.logger.exception("Query failed")  # Auto JSON formatting
```

## ⚙️ Configuration Options

| Setting | Description | Default |
|---------|-------------|---------|
| `LOG_LEVEL` | Minimum log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) | `INFO` |
| `LOG_FILE` | Main log file path (`None` for console only) | `None` |
| `ENABLE_SEVERITY_FILES` | Create separate files for warnings/errors/critical messages | `True` |
| `SEVERITY_FILES_DIR` | Directory for severity-specific files | `"logs/severity"` |
| `MAX_LOG_SIZE_MB` | Max log file size before rotation (MB) | `10` |
| `BACKUP_COUNT` | Number of backup files to keep | `5` |

## 📖 Complete Example

```python
from your_logging_module import Logger, LoggerConfig

# Initialize logger manager
logger_manager = Logger()

# Create different loggers for different components
app_logger = logger_manager.setup_logger(
    logger_name="app.main",
    config=LoggerConfig(LOG_FILE="logs/app.log")
)

db_logger = logger_manager.setup_logger(
    logger_name="app.database", 
    config=LoggerConfig(LOG_FILE="logs/db.log")
)

# Use the loggers
app_logger.info("Application starting")

try:
    db_logger.debug("Connecting to database")
    # Database operations
    db_logger.info("Database connection successful")
except Exception:
    db_logger.exception("Database connection failed")  # JSON formatted

app_logger.info("Application shutting down")

# Clean up
logger_manager.close_all_loggers()
```

## 🔍 Error Formatting

Exceptions are automatically formatted as JSON with complete context:

```json
{
  "error_type": "ZeroDivisionError",
  "error_message": "division by zero",
  "file": "example.py",
  "line": 15,
  "function": "calculate",
  "traceback": "Traceback (most recent call last)...",
  "timestamp": "2023-11-15 10:30:45",
  "level": "ERROR",
  "logger": "main",
  "message": "Calculation failed"
}
```

## 📁 File Structure

The wrapper automatically creates this directory structure:

```
logs/
├── app.log                 # Main application log
├── complete_log.log        # All messages from all loggers
└── severity/               # Severity-based logs
    ├── warning.log        # WARNING messages
    ├── error.log          # ERROR messages
    └── critical.log       # CRITICAL messages
```

## 🔄 Migration from Standard Logging

If you're already using Python's logging:

```python
# Before (standard logging)
import logging
logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)

# After (with this wrapper)
from your_logging_module import Logger
logger_manager = Logger()
logger = logger_manager.setup_logger(
    logger_name=__name__,
    config=LoggerConfig(LOG_FILE="app.log")
)

# All existing logging calls work exactly the same
logger.info("This works like before")
logger.error("But now with added benefits!")
```

## 🎯 Benefits

- **Automatic Setup** - No manual file/directory creation needed
- **Structured Errors** - JSON-formatted exceptions with full context  
- **Visual Differentiation** - Colored console output
- **Flexible Configuration** - Environment variables or programmatic setup
- **Standard Compatible** - Works with existing logging code

## 🛠️ Troubleshooting

**Log files not created?** Check write permissions to the log directory.

**No colors in console?** Ensure you're running in a terminal that supports ANSI colors.

**Missing logs?** Verify your `LOG_LEVEL` setting isn't filtering out messages.

---

> **Note:** Replace `your_logging_module` with your actual module name in the import statements.
