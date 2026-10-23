from grounded_videoqa.gqa_core import sampled_frame_ids, box_iou, expand_box, DenseTrack
from grounded_videoqa.advanced_models import parse_molmo_points


def test_sampled_ids_match_dataset_cadence():
    assert sampled_frame_ids(998) == list(range(0, 998, 30))


def test_geometry():
    assert box_iou([0, 0, 1, 1], [0, 0, 1, 1]) == 1
    b = expand_box([.4, .4, .6, .6], 1.1, 1.1)
    assert b[0] < .4 and b[2] > .6


def test_molmo2_point_format():
    assert parse_molmo_points('<points coords="1 1 752 506">') == [(.752, .506)]
    assert parse_molmo_points('<points coords="752,506">') == [(.752, .506)]
    assert parse_molmo_points('<point x="44.3" y="73.6">') == [(.443, .736)]


def test_track_contract():
    t = DenseTrack(3, [0, 30], [[0, 0, .2, .2], [.1, .1, .3, .3]], [.8, .7])
    assert t.to_submission()["id"] == 3
