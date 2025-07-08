#!/usr/bin/env python3
"""
zip_folder.py ― 指定したフォルダを再帰的に .zip へ圧縮するスクリプト。

Usage
-----
# 例1: フォルダ名と同じ <folder>.zip をカレントに生成
$ python zip_folder.py path/to/folder

# 例2: 出力パスを指定
$ python zip_folder.py path/to/folder -o /tmp/output.zip
"""

import argparse
import os
import pathlib
import zipfile


def zipdir(dir_path: pathlib.Path, zip_path: pathlib.Path, exclude=("env_swe",)) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(dir_path):
            # --- ここで除外フォルダをスキップ ---
            dirs[:] = [d for d in dirs if d not in exclude]
            # ------------------------------------
            for fname in files:
                abs_path = pathlib.Path(root) / fname
                rel_path = abs_path.relative_to(dir_path)
                zf.write(abs_path, arcname=rel_path)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Recursively zip a folder.")
    parser.add_argument("folder", help="圧縮したいフォルダ")
    parser.add_argument(
        "-o",
        "--output",
        help="出力 .zip パス（省略時は <foldername>.zip がカレントに作成）",
        default=None,
    )
    args = parser.parse_args(argv)

    folder = pathlib.Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"{folder} is not a directory")

    zip_path = (
        pathlib.Path(args.output).expanduser().resolve()
        if args.output
        else pathlib.Path.cwd() / f"{folder.name}.zip"
    )

    zipdir(folder, zip_path)
    print(f"✅ Created {zip_path} ({zip_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
