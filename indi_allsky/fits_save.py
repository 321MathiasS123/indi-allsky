"""Immutable FITS snapshots and lossless background archive publication."""
import gzip
import os
from pathlib import Path
import shutil
import tempfile
import zlib


MAX_PENDING = 8
_COPY_CHUNK = 1024 * 1024


def pending_directory(image_dir):
    directory = Path(image_dir).resolve().joinpath('.fits_pending')
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    if directory.resolve() != directory:
        raise ValueError('FITS pending directory must not be a symbolic link')
    return directory


def pending_count(image_dir):
    return sum(path.is_file() for path in pending_directory(image_dir).glob('*.fit'))


def source_path(image_dir, source):
    directory = pending_directory(image_dir)
    path = Path(image_dir).resolve().joinpath(source).resolve()
    if path.parent != directory or path.suffix != '.fit':
        raise ValueError('FITS source must be a snapshot in the pending directory')
    return path


def destination_path(image_dir, filename):
    directory = Path(image_dir).resolve()
    path = directory.joinpath(filename).resolve()
    pending = directory.joinpath('.fits_pending').resolve()
    if (directory not in path.parents or pending in path.parents
            or not path.name.endswith('.fit.gz')):
        raise ValueError('FITS destination must be an archive inside the image directory')
    return path


def _same_contents(source, destination):
    with source.open('rb') as original:
        try:
            with gzip.open(destination, 'rb') as archive:
                while True:
                    expected = original.read(_COPY_CHUNK)
                    if archive.read(_COPY_CHUNK) != expected:
                        return False
                    if not expected:
                        return True
        except (OSError, EOFError, zlib.error):
            return False


def publish(source, destination):
    """Publish a maximum-compression archive without replacing another file.

    A retry can reuse an existing archive only after comparing its complete
    decompressed contents with the immutable source. The caller owns source
    cleanup after its database and upload work succeeds.
    """
    source = Path(source)
    destination = Path(destination)
    if destination.exists():
        if _same_contents(source, destination):
            return destination.stat().st_size
        raise FileExistsError('FITS destination contains different data: {0}'.format(destination))

    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                mode='w+b', prefix='.fits-save-', suffix='.tmp',
                dir=destination.parent, delete=False) as output:
            temporary = Path(output.name)
            with source.open('rb') as original:
                with gzip.GzipFile(filename='', fileobj=output, mode='wb', compresslevel=9) as archive:
                    shutil.copyfileobj(original, archive, length=_COPY_CHUNK)
        temporary.chmod(0o644)
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if not _same_contents(source, destination):
                raise
        return destination.stat().st_size
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
