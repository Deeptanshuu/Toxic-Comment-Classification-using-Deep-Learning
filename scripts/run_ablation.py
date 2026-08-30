"""Control run for the language-conditioning ablation.

The treatment run trains with lang_ids active. This one disables the language
pathway entirely (disable_lang_conditioning), changing nothing else, so the
difference between the two isolates what language conditioning is worth.

Everything else is held identical on purpose, including the fabricated
class_adjustments table in training_config.py. It distorts class weights by
~3.5% on average, but it distorts BOTH arms the same way, so it cancels in the
comparison. Removing it from one arm only would break the ablation.

Writes to its own checkpoint dir and its own MLflow run tagged run.kind=control.
"""
import os
import sys

os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ['TOXIC_DISABLE_LANG_CONDITIONING'] = '1'

from model.training_config import TrainingConfig
import model.train as T


def main():
    cfg = TrainingConfig()
    if not cfg.disable_lang_conditioning:
        sys.exit("refusing to run: disable_lang_conditioning did not take effect")
    cfg.checkpoint_dir = 'weights/toxic_classifier_xlmr_v2_ablation'
    print(f"ABLATION control run: lang conditioning OFF, checkpoints -> {cfg.checkpoint_dir}")
    T.main(config=cfg)


if __name__ == '__main__':
    main()
