"""
Common utility functions
"""
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


def load_config(config_name: str, config_dir: Optional[Path] = None) -> Dict:
    """
    Load YAML configuration file

    Args:
        config_name: Name of config file (e.g., 'paths', 'params_eda')
        config_dir: Directory containing config files. If None, use default location

    Returns:
        Dictionary containing configuration
    """
    if config_dir is None:
        # Default: src/config/
        config_dir = Path(__file__).parent.parent / "config"

    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def parse_filename(filename: str) -> Dict[str, any]:
    """
    Parse filename to extract metadata

    Filename pattern: {product}_Sample{XX} {direction}{run_id}_{datetime}.csv
    Example: "100W_Sample03 ccw4_2025-11-07 03-35-27.csv"

    Args:
        filename: CSV filename

    Returns:
        Dictionary with keys: product, sample, direction, run_id, datetime
        Returns None values if parsing fails
    """
    # Pattern: (100W|200W) Sample{digits} {cw|ccw}{digits} datetime
    pattern = r'(100W|200W)[_\s]*Sample\s*(\d+)\s+(cw|ccw)\s*(\d+)[_\s]*([\d\-\s:]+)\.csv'

    match = re.search(pattern, filename, re.IGNORECASE)

    if match:
        return {
            'product': match.group(1).upper(),  # 100W or 200W
            'sample': int(match.group(2)),      # Sample number
            'direction': match.group(3).upper(),  # CW or CCW
            'run_id': int(match.group(4)),      # Run number
            'datetime': match.group(5).strip(),  # DateTime string
            'filename': filename
        }
    else:
        logging.warning(f"Failed to parse filename: {filename}")
        return {
            'product': None,
            'sample': None,
            'direction': None,
            'run_id': None,
            'datetime': None,
            'filename': filename
        }


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Setup logging configuration

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file. If None, log only to console
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def create_file_id(product: str, sample: int, direction: str, run_id: int) -> str:
    """
    Create unique file ID

    Args:
        product: 100W or 200W
        sample: Sample number
        direction: CW or CCW
        run_id: Run number

    Returns:
        File ID string (e.g., "100W_S00_CW_R4")
    """
    return f"{product}_S{sample:02d}_{direction}_R{run_id}"


def get_project_root() -> Path:
    """
    Get project root directory

    Returns:
        Path to project root
    """
    # Assuming this file is in src/utils/
    return Path(__file__).parent.parent.parent


def resolve_path(path_str: str, base_dir: Optional[Path] = None) -> Path:
    """
    Resolve relative path to absolute path

    Args:
        path_str: Path string (can be relative)
        base_dir: Base directory for relative paths. If None, use project root

    Returns:
        Absolute Path object
    """
    if base_dir is None:
        base_dir = get_project_root()

    path = Path(path_str)

    if not path.is_absolute():
        path = base_dir / path

    return path.resolve()
