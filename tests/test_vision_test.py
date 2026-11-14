"""Production-critical tests for test_image.py."""

from benchmarker.vision_test import VISION_Q, generate_test_image, image_message


class TestTestImage:
    def test_returns_png_bytes(self):
        png, number, colour = generate_test_image(64)
        assert isinstance(png, bytes)
        assert png[:4] == b"\x89PNG"

    def test_returns_4_digit_number(self):
        png, number, colour = generate_test_image(64)
        assert len(number) == 4
        assert number.isdigit()
        assert all(c in "123456789" for c in number)

    def test_returns_colour(self):
        png, number, colour = generate_test_image(64)
        assert colour in ("red", "green", "blue")

    def test_deterministic_with_seed(self):
        png1, num1, col1 = generate_test_image(64, seed=42)
        png2, num2, col2 = generate_test_image(64, seed=42)
        assert num1 == num2
        assert col1 == col2
        assert png1 == png2


class TestImageMessage:
    def test_returns_list_with_user_role(self):
        msg = image_message(b"fake_png", "Describe this")
        assert isinstance(msg, list)
        assert msg[0]["role"] == "user"

    def test_contains_image_url(self):
        msg = image_message(b"\x89PNG\x00", "test")
        content = msg[0]["content"]
        assert any(item.get("type") == "image_url" for item in content)

    def test_contains_text(self):
        msg = image_message(b"\x89PNG\x00", "Describe this")
        content = msg[0]["content"]
        assert any(item.get("type") == "text" for item in content)


class TestVisionQ:
    def test_is_string(self):
        assert isinstance(VISION_Q, str)

    def test_contains_required_fields(self):
        assert "NUMBER=" in VISION_Q
        assert "COLOUR=" in VISION_Q
