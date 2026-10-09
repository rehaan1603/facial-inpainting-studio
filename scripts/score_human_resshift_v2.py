"""Score perceptual pilot using the same frozen metrics, in separate outputs."""
import score_human_resshift_v1 as scorer
from train_human_resshift_v2 import OUT

if __name__=='__main__':
    scorer.OUT=OUT
    scorer.main()
