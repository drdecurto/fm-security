"""
Optional kagglehub loader mixin for SWaT, HAI, and WUSTL-IIoT-2021.

To activate Kaggle-based loading without rewriting the dataset classes,
drop this file into `src/data/` and call `apply_kagglehub_patch()` once
in your driver script:

    from src.data import apply_kagglehub_patch
    apply_kagglehub_patch()

After the patch, the three dataset classes will use `kagglehub` to
auto-download the corresponding mirror on first load and cache it under
`~/.cache/kagglehub/datasets/`. The existing `data/raw/{name}/` fallback
path is preserved when `kagglehub` is unavailable or when the environment
variable OTICS_DATA_BACKEND is set to "local".

Requires: pip install kagglehub
"""

import logging
import os

from src.data.config import SWAT_RAW_DIR, HAI_RAW_DIR, WUSTL_RAW_DIR
from src.data.tabular import swat as _swat
from src.data.tabular import hai as _hai
from src.data.tabular import wustl_iiot as _wustl

logger = logging.getLogger(__name__)

# Mirror identifiers ─────────────────────────────────────────────────────────
KAGGLE_MIRRORS = {
    "swat":  "vishala28/swat-dataset-secure-water-treatment-system",
    "hai":   "icsdataset/hai-security-dataset",
    "wustl": "annaamalaiu/wustl-iiot-2021-dataset",
}


def _kagglehub_available() -> bool:
    try:
        import kagglehub  # noqa: F401
        return True
    except ImportError:
        return False


def _fetch_with_kagglehub(name: str) -> str:
    """Download the Kaggle mirror for `name` and return the local cache path."""
    import kagglehub
    handle = KAGGLE_MIRRORS[name]
    logger.info(f"[kagglehub] downloading {handle} (this may take a moment on first use)...")
    return kagglehub.dataset_download(handle)


def _patched_swat_raw_dir() -> str:
    """SWaT raw-dir resolver: prefer Kaggle mirror unless OTICS_DATA_BACKEND='local'."""
    if os.environ.get("OTICS_DATA_BACKEND", "kagglehub").lower() == "local":
        return SWAT_RAW_DIR
    if not _kagglehub_available():
        logger.warning("[kagglehub] not installed; falling back to local SWAT_RAW_DIR")
        return SWAT_RAW_DIR
    return _fetch_with_kagglehub("swat")


def _patched_hai_raw_dir() -> str:
    if os.environ.get("OTICS_DATA_BACKEND", "kagglehub").lower() == "local":
        return HAI_RAW_DIR
    if not _kagglehub_available():
        logger.warning("[kagglehub] not installed; falling back to local HAI_RAW_DIR")
        return HAI_RAW_DIR
    return _fetch_with_kagglehub("hai")


def _patched_wustl_raw_dir() -> str:
    if os.environ.get("OTICS_DATA_BACKEND", "kagglehub").lower() == "local":
        return WUSTL_RAW_DIR
    if not _kagglehub_available():
        logger.warning("[kagglehub] not installed; falling back to local WUSTL_RAW_DIR")
        return WUSTL_RAW_DIR
    return _fetch_with_kagglehub("wustl")


def apply_kagglehub_patch() -> None:
    """
    Replace the static SWAT_RAW_DIR / HAI_RAW_DIR / WUSTL_RAW_DIR constants
    referenced inside the loader modules with the dynamic resolvers above.

    This is a one-shot, idempotent operation. Call it once at the top of a
    driver script before constructing any dataset instance.
    """
    if getattr(apply_kagglehub_patch, "_applied", False):
        return

    _swat.SWAT_RAW_DIR = _patched_swat_raw_dir()
    _hai.HAI_RAW_DIR = _patched_hai_raw_dir()
    _wustl.WUSTL_RAW_DIR = _patched_wustl_raw_dir()

    apply_kagglehub_patch._applied = True
    logger.info("[kagglehub] patch applied to SWAT/HAI/WUSTL loaders")
