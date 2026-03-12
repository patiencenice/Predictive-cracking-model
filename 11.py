import os
from pathlib import Path


def batch_rename(
    folder: str,
    prefix: str,
    dry_run: bool = True,
) -> None:
    """
    批量给文件添加前缀。

    :param folder: 目标文件夹路径
    :param prefix: 要添加的前缀，比如 "new_"
    :param dry_run: 为 True 时只打印不真正修改
    """
    folder_path = Path(folder)

    if not folder_path.exists() or not folder_path.is_dir():
        print(f"目录不存在或不是目录: {folder_path}")
        return

    files = [p for p in folder_path.iterdir() if p.is_file()]
    if not files:
        print("该目录下没有文件。")
        return

    print(f"目标目录: {folder_path.resolve()}")
    print(f"前缀: {prefix!r}")
    print(f"预览模式(dry_run): {dry_run}")
    print("-" * 40)

    for file_path in files:
        new_name = prefix + file_path.name
        new_path = file_path.with_name(new_name)

        print(f"{file_path.name}  ->  {new_name}")
        if not dry_run:
            file_path.rename(new_path)

    print("-" * 40)
    if dry_run:
        print("当前为预览模式，没有真正重命名。")
        print("确认无误后，可将 dry_run=False 再运行一次。")
    else:
        print("重命名完成。")


if __name__ == "__main__":
    # 在这里修改成你自己的目录和前缀
    target_folder = r"d:\cursor\code"   # 例如：r"d:\cursor\code"
    prefix_text = "new_"
    # 先预览
    batch_rename(target_folder, prefix_text, dry_run=True)
    # 如果预览没问题，再执行真正重命名：
    # batch_rename(target_folder, prefix_text, dry_run=False)