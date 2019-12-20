import os
import shutil
import glob
import re
from bs4 import BeautifulSoup


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
            with open(html_file, mode="r", encoding="utf-8") as fp:
                doc = fp.read()
                soup = BeautifulSoup(doc, features="html.parser")

            script_tags = soup.find_all("script")
            for tag in script_tags:
                tag_src = tag.get("src")
                if not tag_src:
                    continue
                if not tag_src.startswith("_static"):
                    continue
                new_tag_src = "docs/{}".format(tag_src)
                tag.attrs['src'] = new_tag_src

            link_tags = soup.find_all("link")
            for tag in link_tags:
                tag_href = tag.get('href')
                if not tag_href:
                    continue
                if not tag_href.startswith("_static"):
                    continue
                new_tag_href = "docs/{}".format(tag_href)
                tag.attrs['href'] = new_tag_href

            with open(file_path, mode="w+", encoding="utf-8") as fp:
                fp.write(soup.prettify())

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
