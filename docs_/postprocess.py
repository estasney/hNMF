import os
import shutil
import glob


def main(dry_run):
    cwd = os.getcwd()
    parent_folder = os.path.split(cwd)[0]
    target_folder = os.path.join(parent_folder, "docs")

    source_build = os.path.join(cwd, "build", "singlehtml")
    source_html = glob.glob(os.path.join(source_build, "*.html"))
    source_static = os.path.join(source_build, "_static")

    for html_file in source_html:
        file_name = os.path.split(html_file)[1]
        file_path = os.path.join(target_folder, file_name)

        if dry_run:
            print("Copy {} to {}".format(html_file, file_path))
        else:
            shutil.copyfile(html_file, file_path)

    # delete static destination

    target_folder_static = os.path.join(target_folder, "_static")

    if os.path.isdir(target_folder_static):
        if dry_run:
            print("Delete static at {}".format(target_folder_static))
        else:
            shutil.rmtree(target_folder_static)

    # copy static to destination

    if dry_run:
        print("Copy static from {} to {}".format(source_static, target_folder_static))
    else:
        shutil.copytree(source_static, target_folder_static)


if __name__ == '__main__':
    main(dry_run=False)
