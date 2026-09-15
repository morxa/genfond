from genfond.shutdown import request_stop, reset_stop, stop_requested


def test_stop_requested_is_false_until_requested():
    reset_stop()
    try:
        assert stop_requested() is False
        request_stop()
        assert stop_requested() is True
    finally:
        reset_stop()


def test_reset_stop_clears_a_previous_request():
    request_stop()
    reset_stop()
    assert stop_requested() is False
