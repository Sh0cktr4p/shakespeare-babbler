
import hydra

from shakespeare_babbler.utils import TransformerTrainer


@hydra.main(config_path='config', config_name='shakespeare-babbler', version_base=None)
def main(cfg):
    trainer = TransformerTrainer.from_config(cfg)
    trainer.train(cfg.training.n_iterations)


if __name__ == '__main__':
    main()
