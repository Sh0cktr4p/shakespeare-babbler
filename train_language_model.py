
import hydra

from utils import TransformerTrainer


@hydra.main(config_path='config', config_name='shakespeare-babbler')
def main(cfg):
    trainer = TransformerTrainer.from_config(cfg)
    trainer.train(cfg.training.n_iterations)


if __name__ == '__main__':
    main()
