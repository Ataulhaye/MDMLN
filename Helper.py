from pathlib import Path


class Helper:

    @staticmethod
    def ensure_dir(root_dir, directory):
        directory_path = Path(f"{root_dir}/{directory}").absolute().resolve()
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return directory_path
