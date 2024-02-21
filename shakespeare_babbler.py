import time
from threading import Thread

import hydra

from utils import TransformerTrainer


def text_iterator(trainer: TransformerTrainer, n_tokens, context=None):
    buffer = [context, None]

    def get_next_tokens(buffer_index):
        context = trainer._get_text(
            n_tokens=n_tokens,
            context=buffer[buffer_index % 2],
        )
        buffer[(buffer_index + 1) % 2] = context

    i = 0
    get_next_tokens(i)

    while True:
        i += 1
        thread = Thread(target=get_next_tokens, args=(i,))
        thread.start()

        for t in range(n_tokens):
            yield buffer[i % 2][-n_tokens + t]
        thread.join()


@hydra.main(config_path="config", config_name="shakespeare-babbler")
def main(cfg):
    trainer = TransformerTrainer.from_config(cfg, load_state_dict=True)

    print("Loading model complete!")

    for text in text_iterator(
        trainer=trainer,
        n_tokens=8,
    ):
        print(text, end="", flush=True)
        time.sleep(0.03)


if __name__ == '__main__':
    main()
