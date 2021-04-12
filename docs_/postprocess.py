import os
import shutil
import glob

from bs4 import BeautifulSoup

def main():
    cwd = os.getcwd()
    parent_folder = os.path.split(cwd)[0]
    target_folder = os.path.join(parent_folder, "docs")

    source_build = os.path.join(cwd, "build", "singlehtml")
    source_html = glob.glob(os.path.join(source_build, "*.html"))
    source_static = os.path.join(source_build, "_static")

    for html_file in source_html:
        file_name = os.path.split(html_file)[1]
        file_path = os.path.join(target_folder, file_name)

        with open(html_file, mode="r", encoding="utf-8") as fp:
            doc = fp.read()
            soup = BeautifulSoup(doc, features="html.parser")

        with open(file_path, mode="w+", encoding="utf-8") as fp:
            fp.write(soup.prettify())

        print(f"Wrote {file_name} to {file_path}")

    target_folder_static = os.path.join(target_folder, "_static")
    print(target_folder_static)
    if os.path.exists(target_folder_static):
        print("Removing Static Folder from Docs")
        shutil.rmtree(target_folder_static)

    shutil.copytree(source_static, target_folder_static)


if __name__ == '__main__':
    main()
