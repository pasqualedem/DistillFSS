import os
import click


PREFIX = "wandb: "

@click.command()
@click.argument('path', type=click.Path(exists=True))
def sync_folder(path):
    print(f'Syncing folder {path}')
    files = os.listdir(path)
    out_files = [f for f in files if f.endswith('.log') and not f.startswith('grid')]
    failed_files = []
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
                failed_files.append(f)
                continue
            sync_line = sync_lines[0]
            print(sync_line[len(PREFIX):].strip())
    
    # Read grid.log and find lines corresponding to failed files
    grid_log_path = os.path.join(path, 'grid.log')
    if os.path.exists(grid_log_path):
        with open(grid_log_path) as grid_file:
            grid_lines = grid_file.readlines()
            for failed_file in failed_files:
                for line in grid_lines:
                    if failed_file in line:
                        print(f'Failed file {failed_file} command: {line.strip()}')
    else:
        print(f'grid.log not found in {path}')


if __name__ == "__main__":
    sync_folder()
            