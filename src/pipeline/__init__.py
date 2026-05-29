"""Amazon fashion ETL pipeline (Phase 2).

Run as a module:  python -m src.pipeline

Note: we intentionally do NOT re-export run() here. Importing it would pull in
the heavy embed path (torch/SBERT); keeping __init__ light means the API can
import pipeline.config without dragging in those deps at serve time.
"""
