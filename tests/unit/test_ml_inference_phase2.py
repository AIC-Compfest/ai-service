from app.services.extraction.ml.decode import decode


def test_decode_abstains_and_respects_many_cardinality():
    candidates = {"total": [({"m_candidate_id": "a", "m_parsed_value": "10", "m_page": "1"}, .95)], "tag": [({"m_candidate_id": "b", "m_parsed_value": "x", "m_page": "1"}, .8), ({"m_candidate_id": "c", "m_parsed_value": "y", "m_page": "1"}, .7)]}
    result = decode(candidates, {"total": {"cardinality": "one"}, "tag": {"cardinality": "many"}}, {"total": {"t_unknown": .4, "t_auto": .9}, "tag": {"t_unknown": .75, "t_auto": .9}})
    assert result["total"]["status"] == "accepted"
    assert len(result["tag"]["candidates"]) == 1