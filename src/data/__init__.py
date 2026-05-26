"""Dataset loaders for OT/ICS intrusion-detection benchmarks."""

from src.data.tabular.swat import SWaTDataset
from src.data.tabular.hai import HAIDataset
from src.data.tabular.wustl_iiot import WUSTLIIoTDataset

# Optional: kagglehub auto-download patch. Activate explicitly by calling
# `apply_kagglehub_patch()` once in your driver script.
try:
    from src.data.kagglehub_loader import apply_kagglehub_patch  # noqa: F401
    _has_kagglehub_patch = True
except ImportError:
    _has_kagglehub_patch = False

__all__ = ["SWaTDataset", "HAIDataset", "WUSTLIIoTDataset"]
if _has_kagglehub_patch:
    __all__.append("apply_kagglehub_patch")
