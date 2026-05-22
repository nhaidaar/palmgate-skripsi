def test_opencv_camera_source_opens_device_path(monkeypatch):
    import app.camera as camera

    opened_sources = []

    class FakeCapture:
        def __init__(self, source):
            opened_sources.append(source)

        def isOpened(self):
            return True

    monkeypatch.setattr(camera.cv2, "VideoCapture", FakeCapture)

    camera.OpenCVCameraSource("/dev/video1")

    assert opened_sources == ["/dev/video1"]
