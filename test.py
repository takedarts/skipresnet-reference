import pytest
import timm
import torch
import models  # noqa

MODELS = [
    ('skipresnet34', 1.0498065948486328),
    ('skipresnet50', 2.5082178115844727),
    ('skipresnet101', 2.6629934310913086),
    ('skipresnext50_32x4d', 2.590874195098877),
    ('skipresnext101_32x4d', 3.639589309692383),
]


@pytest.mark.parametrize(
    'model_name,target_value',
    MODELS,
    ids=[n for n, _ in MODELS])
def test_model(model_name: str, target_value: float) -> None:
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    with torch.no_grad():
        output = model(torch.ones([1, 3, 256, 256], dtype=torch.float32))
        output_value = float(output.sum().abs())

    assert abs(output_value - target_value) < 1e-3


if __name__ == '__main__':
    pytest.main(['-v', __file__])
