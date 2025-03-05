import os
import click


PREFIX = "wandb: "

@click.command()
@click.argument('path', type=click.Path(exists=True))
def sync_folder(path):
    print(f'Syncing folder {path}')
    files = os.listdir(path)
    out_files = [f for f in files if f.endswith('.log') and not f.startswith('grid')]
    for f in out_files:
        with open(os.path.join(path, f)) as file:
            # Grep line with "wandb sync" command
            try:
                lines = file.readlines()
            except UnicodeDecodeError:
                print(f'Error reading {os.path.join(path, f)}')
                continue
            sync_lines = [l for l in lines if 'wandb sync' in l]
            if len(sync_lines) == 0:
                print(f'No sync command found in {os.path.join(path, f)}')
                continue
            sync_line = sync_lines[0]
            print(sync_line[len(PREFIX):].strip())


if __name__ == "__main__":
    sync_folder()
            