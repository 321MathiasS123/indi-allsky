"""Exercise gallery panorama matching without the capture host's D-Bus stack."""

import ast
import logging
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import Column, DateTime, Integer, JSON, String, and_, create_engine, null, or_
from sqlalchemy.orm import Session, declarative_base


ROOT = Path(__file__).resolve().parents[2]
Base = declarative_base()


class Asset:
    id = Column(Integer, primary_key=True)
    camera_id = Column(Integer)
    createDate = Column(DateTime)
    remote_url = Column(String)
    s3_key = Column(String)
    width = 100
    height = 50

    def getUrl(self, s3_prefix='', local=True):
        return self.remote_url or f'{s3_prefix}/{self.id}.jpg'


class Image(Asset, Base):
    __tablename__ = 'image'
    thumbnail_uuid = Column(String)
    createDate_year = Column(Integer)
    createDate_month = Column(Integer)
    createDate_day = Column(Integer)
    createDate_hour = Column(Integer)
    detections = Column(Integer, default=0)
    data = Column(JSON)
    exclude = False


class Thumbnail(Asset, Base):
    __tablename__ = 'thumbnail'
    uuid = Column(String)


class Panorama(Asset, Base):
    __tablename__ = 'panorama'


@pytest.mark.parametrize('local', [True, False])
def test_gallery_matches_camera_and_capture_and_keeps_images_without_panorama(local):
    source = ROOT / 'indi_allsky/flask/forms.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    gallery = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'IndiAllskyGalleryViewer')
    method = next(node for node in gallery.body if isinstance(node, ast.FunctionDef)
                  and node.name == 'getImages')
    engine = create_engine('sqlite://')
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        start = datetime(2026, 9, 27, 0, 0)
        for image_id in range(1, 6):
            captured = start + timedelta(seconds=image_id)
            session.add(Image(id=image_id, camera_id=1, createDate=captured,
                              createDate_year=2026, createDate_month=9, createDate_day=27,
                              createDate_hour=0, thumbnail_uuid=str(image_id),
                              remote_url=f'https://images.example/{image_id}.jpg'))
            session.add(Thumbnail(id=image_id, uuid=str(image_id), camera_id=1))
        session.add_all([
            Panorama(id=11, camera_id=1, createDate=start + timedelta(seconds=1)),
            Panorama(id=12, camera_id=1, createDate=start + timedelta(seconds=2),
                     remote_url='https://images.example/panorama.jpg'),
            Panorama(id=13, camera_id=1, createDate=start + timedelta(seconds=3),
                     s3_key='panorama.jpg'),
            Panorama(id=14, camera_id=2, createDate=start + timedelta(seconds=4),
                     remote_url='https://images.example/other-camera.jpg'),
            Panorama(id=15, camera_id=1, createDate=start + timedelta(seconds=6)),
        ])
        session.commit()
        namespace = dict(db=SimpleNamespace(session=session), and_=and_, or_=or_, sa_null=null,
                         app=SimpleNamespace(logger=logging.getLogger(__name__)),
                         IndiAllSkyDbImageTable=Image, IndiAllSkyDbThumbnailTable=Thumbnail,
                         IndiAllSkyDbPanoramaImageTable=Panorama)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
        viewer = SimpleNamespace(camera_id=1, detections_count=0, local=local, s3_prefix='',
                                 _apply_asi676mc_status_filter=lambda query: query)
        images = namespace['getImages'](viewer, 2026, 9, 27, 0)
        assert [image['id'] for image in images] == [5, 4, 3, 2, 1]
        assert [image['panorama_id'] for image in images] == [None, None, 13, 12, 11 if local else None]
    engine.dispose()


def test_gallery_panorama_toolbar():
    node = shutil.which('node')
    assert node is not None, 'Node.js is required to run the gallery toolbar tests'
    result = subprocess.run(
        [node, '--test', str(Path(__file__).with_name('gallery_panorama.test.cjs'))],
        capture_output=True, text=True, encoding='utf-8', timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
