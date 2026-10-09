"""Use the frozen masked evaluator on explicitly labelled guidance arms."""
from pathlib import Path
import score_masked_sdxl_v1 as scorer
if __name__=='__main__':
    scorer.OUT=Path(__file__).resolve().parents[1]/'outputs/focused_eye_guidance_replication_v2'
    scorer.main()
