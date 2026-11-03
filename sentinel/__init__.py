"""Sentinel: a small ML firewall that flags prompt-injection attempts."""
from .model import Verdict, check, evaluate, load_model, save_model, train, tune_threshold

__all__ = ["Verdict", "check", "evaluate", "load_model", "save_model", "train", "tune_threshold"]
