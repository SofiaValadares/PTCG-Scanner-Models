"""python -m cropper photo.jpg [--preview]  |  python -m cropper export FOLDER"""

from __future__ import annotations

import sys


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "export":
        del sys.argv[1]
        from .export import main as export_main

        return export_main()
    from .crop_from_obb import main as crop_main

    return crop_main()


if __name__ == "__main__":
    raise SystemExit(main())
