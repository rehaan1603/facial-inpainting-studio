"""Reuse the frozen masked metrics with labelled local/global controls."""
from pathlib import Path
import score_masked_sdxl_v1 as scorer
if __name__=='__main__':
    scorer.OUT=Path(__file__).resolve().parents[1]/'outputs/focused_eye_local_appearance_v1'
    scorer.main()
