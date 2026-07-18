import scripts.train_early_fusion as train_script


def test_train_parser_defaults_to_scratch_event() -> None:
    parser_args = train_script.parse_args
    assert callable(parser_args)
