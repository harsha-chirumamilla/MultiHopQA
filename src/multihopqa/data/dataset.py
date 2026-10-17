"""Read prepared JSONL examples without keeping the whole file in memory."""

import json

from torch.utils.data import Dataset


class HotpotJsonlDataset(Dataset):
    """A small random-access reader for the prepared HotpotQA JSONL files."""

    def __init__(self, file_path):
        self.file_path = str(file_path)
        self.line_offsets = []

        with open(self.file_path, "rb") as input_file:
            while True:
                line_start = input_file.tell()
                line = input_file.readline()
                if not line:
                    break
                self.line_offsets.append(line_start)

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index):
        with open(self.file_path, "rb") as input_file:
            input_file.seek(self.line_offsets[index])
            line = input_file.readline().decode("utf-8")
        return json.loads(line)
