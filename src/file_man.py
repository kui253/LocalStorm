import os
from langchain.schema import Document


class FileObject:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    @property
    def content(self):
        p_dir = os.path.dirname(self.path)
        p_dir = os.path.dirname(p_dir)
        summary_dir = os.path.join(p_dir, "summary", self.name)
        with open(self.path, "r", encoding="utf-8") as file, open(
            summary_dir, "r", encoding="utf-8"
        ) as file2:
            text_content = file.read()
            summary_content = file2.read()
            title = self.name.split(".")[0]

            return Document(
                metadata={
                    "description": "no desc",
                    "title": title,
                    "snippets": text_content,
                    "url": self.path,
                },
                page_content=summary_content,
            )

    @property
    def summary(self):
        p_dir = os.path.dirname(self.path)
        p_dir = os.path.dirname(p_dir)
        summary_dir = os.path.join(p_dir, "summary", self.name)
        with open(summary_dir, "r", encoding="utf-8") as file:
            return Document(page_content=file.read())

    def to_dict(self):
        return {"name": self.name, "content": self.content}


class FileManager:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.visited = False
        self.subfolders = {}
        self.files = []
        self._populate_file_structure()

    def _populate_file_structure(self):

        for root, dirs, files in os.walk(self.root_dir):
            relative_path = os.path.relpath(root, self.root_dir)
            if relative_path == ".":
                current_folder = self
            else:
                current_folder = self._get_folder_from_path(relative_path)
            if not current_folder.visited:
                for dir_name in dirs:
                    dir_path = os.path.join(root, dir_name)
                    current_folder.subfolders[dir_name] = FileManager(dir_path)

                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    file_object = FileObject(file_name, file_path)
                    current_folder.files.append(file_object.to_dict())
                current_folder.visited = True

    def _get_folder_from_path(self, path):
        parts = path.split(os.sep)
        current_folder = self
        for part in parts:
            current_folder = current_folder.subfolders[part]
        return current_folder

    def __getitem__(self, key):
        return self.subfolders[key]

    def __repr__(self):
        return f"FileManager(subfolders={list(self.subfolders.keys())}, files={self.files})"

    def print_structure(self, indent=0):
        prefix = "-" * indent
        print(f"{prefix}{os.path.basename(self.root_dir)}/")
        indent += 4
        prefix = "-" * indent
        for file in self.files:
            print(f"{prefix}{file['name']}")
        for subfolder_name, subfolder in self.subfolders.items():
            subfolder.print_structure(indent)

    def get_all_files(self, all_files=[]):
        for file in self.files:
            all_files.append(file)
        for subfolder_name, subfolder in self.subfolders.items():
            subfolder.get_all_files(all_files)

        return all_files


# 使用示例
# 假设文件夹结构如下：
# /root_dir
# ├── subfolder1
# │   ├── subsubfolder1
# │   │   ├── file1.txt
# │   │   └── file2.txt
# │   └── file3.txt
# └── subfolder2
#     └── file4.txt

# 初始化FileManager对象
# root_dir = "/data2/whd/storm/database/methodSdata/data/txt"
# file_manager = FileManager(root_dir)
# file_manager.print_structure(4)
# all_files = []
# file_manager.get_all_files(all_files)
# import pdb

# pdb.set_trace()
# print(len(all_files))
