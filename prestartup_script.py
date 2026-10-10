"""Honor an explicit local opt-in to native Assets, without editing ComfyUI core."""
import json
import logging
from pathlib import Path
import sys
import folder_paths
from comfy.cli_args import args

try:
    preference=Path(folder_paths.get_user_directory())/'h3_fullstack'/'assets.json'
    if preference.is_file() and json.loads(preference.read_text(encoding='utf-8')).get('enable_assets') is True:
        if hasattr(args,'enable_assets'):
            args.enable_assets=True
            if '--enable-assets' not in sys.argv: sys.argv.append('--enable-assets')
            logging.info('H3 FullStack: native Assets enabled by local user preference')
except (OSError,ValueError,AttributeError):
    logging.warning('H3 FullStack: could not read local Assets preference; command-line settings preserved')
