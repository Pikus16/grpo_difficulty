from src_utils import do_single_run
import click

@click.command()
@click.option(
    '--model-name', '-m',
    default='unsloth/Qwen3-4B-unsloth-bnb-4bit',
    show_default=True,
    help="Model name or path"
)
@click.option(
    '--split',
    default='train',
    show_default=True,
    help="Which split to run on"
)
@click.option('--dataset_name', type=str, required=True)
@click.option('--batch_size', '-b', type=int, default=16, help='Batch size to use')
@click.option('--num_repeat',  type=int, default=10, help='Number of answers per question')
def main(
    model_name: str,
    split: str,
    dataset_name: str,
    batch_size: int,
    num_repeat: int,
):
    assert split in ['train','test']
    print(f'Running pretrained')
    pretrained_accuracy, pretrained_passes = do_single_run(
        model_name=model_name,
        adapter_name=None,
        split=split,
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_repeat=num_repeat
    )
    print(f"Base: Accuracy: {pretrained_accuracy:0.3f}, Pass@{num_repeat}: {pretrained_passes:0.3f}")

if __name__ == '__main__':
    main()