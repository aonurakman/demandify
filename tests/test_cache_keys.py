from demandify.cache.keys import traffic_key, matching_key, bbox_key, network_key, osm_key


def test_traffic_key_includes_zoom_and_style():
    bbox = bbox_key(1.0, 2.0, 3.0, 4.0)
    key1 = traffic_key(bbox, provider="tomtom", timestamp_bucket="2024-01-01T12:00", zoom=12, style="absolute")
    key2 = traffic_key(bbox, provider="tomtom", timestamp_bucket="2024-01-01T12:00", zoom=14, style="absolute")
    key3 = traffic_key(bbox, provider="tomtom", timestamp_bucket="2024-01-01T12:00", zoom=12, style="relative")

    assert key1 != key2
    assert key1 != key3


def test_matching_key_combines_inputs():
    bbox = bbox_key(1.0, 2.0, 3.0, 4.0)
    osm = osm_key(bbox)
    net = network_key(osm, car_only=True, seed=42)
    m1 = matching_key(bbox, net, "tomtom", "2024-01-01T12:00")
    m2 = matching_key(bbox, net, "tomtom", "2024-01-01T12:05")

    assert m1 != m2
